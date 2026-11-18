"""Matching by appearance instead of by label string.

The whole pipeline answers "is this the target" with `label == target`, four
times over: which tracks may become candidates (objects/object_layer.candidates),
which live detection stops an approach (agent/nav_agent._best_target_detection),
which tracks a refutation retires (agent/candidate._without_stale_twins,
eval/attempts), and which track anchors the search prior
(exploration/strategy._last_known_target_xy). A string match is a hard gate: the
scissors that the mapping pass called `bleach bottle` is not a near-miss in that
scheme, it is absent.

This module is the other question -- how much does this thing LOOK like the
words of the query -- and the three places it is asked:

  **The container prior.** A surface inherits the appearance of what is resting
  on it, which is DualMap's `related_objs` with our support relation in place of
  its neighbour graph. Scored by the best child, fused into the existing
  affordance x affinity x proximity product.

  **Candidate admission.** A track whose feature matches the query text may be
  proposed even though its label does not. It changes only the label TEST; the
  observation-count, evidence, presence and identity gates are untouched, and
  nothing here may stop an approach on its own.

  **The fallback after a refutation.** A child the agent has been to and not
  found stops lending its appearance to its surface.

Two design notes worth keeping, both learned from the cosines themselves.

`feature_term` is not the cosine. CLIP text-image similarities live in a narrow
band (roughly 0.2-0.35 for ViT-B/32), so using one as a multiplicative weight
would compress every surface toward the same number while still being scale-
sensitive in the wrong place. What carries information is the GAP to the best
candidate, so the term is `exp(-beta * (s_max - s))` with beta set from the
measured gap and a floor under it -- a surface no feature speaks for is
ordinary, never impossible. That is the same shape the room bonus uses one level
up (exploration/strategy._room_bonus), and for the same reason.

Similarities are rounded before they meet a threshold. cuBLAS picks kernels by
batch shape, so the same crop can score 0.30001 in one batch and 0.29999 in
another; the runs on this benchmark are exactly reproducible and an admission
that flips on the last decimal would quietly break that.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# Rounding applied to every similarity before comparison or thresholding.
SIM_DECIMALS = 4


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine of two unit features, rounded. -1.0 when either is missing."""
    if a is None or b is None:
        return -1.0
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return -1.0
    return round(float(np.dot(va, vb)), SIM_DECIMALS)


def merge_running_mean(
    old_ft: Optional[np.ndarray], n: int, new_ft: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], int]:
    """Track feature as the renormalised mean of its detections' features.

    DualMap merges the same way (utils/object.py:367-375). A mean rather than
    the best-scoring view because the detector's confidence is about the label
    it chose, which is precisely the channel this module does not trust.
    """
    if new_ft is None:
        return old_ft, int(n)
    new = np.asarray(new_ft, dtype=np.float32).ravel()
    if old_ft is None or int(n) <= 0:
        merged = new
    else:
        old = np.asarray(old_ft, dtype=np.float32).ravel()
        if old.size != new.size:
            return old_ft, int(n)
        merged = (old * float(n) + new) / float(n + 1)
    norm = float(np.linalg.norm(merged))
    if norm <= 1e-8:
        return old_ft, int(n)
    return (merged / norm).astype(np.float32), int(n) + 1


def blend_label_text(
    image_ft: Optional[np.ndarray], label_ft: Optional[np.ndarray], weight: float
) -> Optional[np.ndarray]:
    """DualMap's weighted feature: 0.7 * image + 0.3 * text-of-detected-label,
    renormalised (config/system_config.yaml:101-102).

    Off by default. It helps when the detector's label is right and hurts in
    exactly the case this module exists for -- dragging a mis-named scissors
    toward `bleach bottle` is the bug, not the fix -- so it is a parity knob for
    reproducing DualMap's behaviour, not a default.
    """
    if image_ft is None or label_ft is None or weight <= 0.0:
        return image_ft
    w = float(min(max(weight, 0.0), 1.0))
    mixed = (1.0 - w) * np.asarray(image_ft, dtype=np.float32).ravel() + w * np.asarray(
        label_ft, dtype=np.float32
    ).ravel()
    norm = float(np.linalg.norm(mixed))
    return image_ft if norm <= 1e-8 else (mixed / norm).astype(np.float32)


def feature_term(
    sim: Optional[float], sim_max: float, beta: float, floor: float
) -> float:
    """The multiplier a container's appearance earns: `floor + (1-floor) *
    exp(-beta * (sim_max - sim))`.

    1.0 for the best-looking surface, decaying with the gap to it, never below
    `floor`. `beta = 0` makes every scored surface 1.0, which is the off switch;
    a surface with no feature at all gets `floor`, because "nothing here looks
    like it" is weaker evidence than "something over there looks much more like
    it" -- the map's crops are one view of one moment, and half the objects in a
    house were never mapped at all.
    """
    if sim is None:
        return float(floor)
    b = max(float(beta), 0.0)
    f = float(min(max(floor, 0.0), 1.0))
    if b <= 0.0:
        return 1.0
    gap = max(float(sim_max) - float(sim), 0.0)
    return float(f + (1.0 - f) * math.exp(-b * gap))


def _presence(track: Any) -> float:
    """A track's belief that its object is still there, or 1.0 when the presence
    filter is off -- an absent belief must not silently veto a candidate."""
    presence = getattr(track, "presence", None)
    p = getattr(presence, "p", None)
    return 1.0 if p is None else float(p)


def refuted(track: Any) -> bool:
    """Has the agent been to this track and found the target was not there?

    The same predicate the stale-twin rule keys on (agent/candidate.py): an
    absence arrival or a scored failed attempt REFUTES a track in place, while
    an unreachable verdict or a low belief does not.
    """
    if track is None:
        return True
    return bool(
        int(getattr(track, "absence_arrivals", 0)) > 0
        or int(getattr(track, "failed_attempts", 0)) > 0
        or bool(getattr(track, "blacklisted", False))
        or bool(getattr(track, "disabled", False))
    )


def admits(
    track: Any, threshold: float, admit_prior_tracks: bool = False
) -> bool:
    """May this track be proposed on appearance alone?

    A prior-map track this episode has never seen is held back unless
    `admit_prior_tracks` is on, and that is a separate arm rather than a default:
    admitting one means walking to where a differently-named object was mapped
    in a previous session, which is the strongest version of the mechanism and
    also the one that can spend an attempt on a year-old crop.
    """
    if track is None:
        return False
    sim = float(getattr(track, "feature_sim", -1.0))
    if sim < float(threshold):
        return False
    if bool(getattr(track, "from_prior", False)) and not bool(getattr(track, "seen_live", False)):
        return bool(admit_prior_tracks)
    return True


def container_children(
    scene_graph, *, floor_key: Optional[int] = None, radius_m: float = 0.0
) -> Dict[int, set]:
    """{container id: the track ids whose appearance it inherits}.

    The support relation first -- our `container_id` is the same "resting on"
    test DualMap's neighbour graph encodes -- and then, when `radius_m` is
    positive, any mapped object within that distance of the surface's centre.

    The widening is not tidiness, it is what the maps say. Over the three prior
    maps only 25/689, 8/343 and 20/520 objects are BOUND to a surface: the
    support test needs the object's base within 0.15 m of the surface top and
    inside its footprint, and a track whose ellipsoid is fitted from a handful
    of distant observations misses that band far more often than the object
    misses the table. Two of the four objects this mechanism exists for -- the
    00880 scissors and the 00848 mug -- have a track within 0.04 m of them and
    no container at all, so a strictly bound child set would score every surface
    near them at the floor and the mechanism could not fire where it was aimed.
    """
    children: Dict[int, set] = {}
    containers = getattr(scene_graph, "containers", {}) or {}
    for cid, node in containers.items():
        ids = {int(t) for t in (getattr(node, "track_ids", []) or [])}
        children[int(cid)] = ids
    for obj in getattr(scene_graph, "objects", []) or []:
        if floor_key is not None and int(getattr(obj, "floor_id", 0)) != int(floor_key):
            continue
        cid = getattr(obj, "container_id", None)
        if cid is not None:
            children.setdefault(int(cid), set()).add(int(obj.track_id))
        if radius_m <= 0.0:
            continue
        centre = np.asarray(getattr(obj, "center", None), dtype=float)
        if centre is None or centre.size < 3:
            continue
        for other, node in containers.items():
            if floor_key is not None and int(
                getattr(node, "floor_id", getattr(node, "floor", 0))
            ) != int(floor_key):
                continue
            node_c = np.asarray(node.center, dtype=float)
            if float(np.hypot(centre[0] - node_c[0], centre[2] - node_c[2])) <= radius_m:
                children.setdefault(int(other), set()).add(int(obj.track_id))
    return children


def container_feature_scores(
    scene_graph,
    object_layer,
    text_ft: Optional[np.ndarray],
    *,
    skip_refuted: bool = True,
    floor_key: Optional[int] = None,
    radius_m: float = 0.0,
) -> Dict[int, float]:
    """{container id: best cosine over what it holds and what it is}.

    DualMap's anchor score is `max(cos(query, anchor), max cos(query, related))`
    -- the bed scores for `scissors` because of the small thing that sat on it.
    The members' own features are included for the same reason the anchor's own
    feature is in DualMap's max: a table that holds nothing mapped is still a
    table, and a query for one should still find it.

    Children the agent has already refuted are dropped, so the surface stops
    being recommended by the object that turned out not to be the target -- the
    appearance half of what `InspectionLog` does for visits.
    """
    scores: Dict[int, float] = {}
    if text_ft is None:
        return scores
    containers = getattr(scene_graph, "containers", {}) or {}
    if not containers:
        return scores
    children = container_children(scene_graph, floor_key=floor_key, radius_m=radius_m)
    for cid, node in containers.items():
        if floor_key is not None and int(getattr(node, "floor_id", getattr(node, "floor", 0))) != int(floor_key):
            continue
        best: Optional[float] = None
        for tid in children.get(int(cid), ()):  # noqa: B007
            track = object_layer.get(int(tid)) if object_layer is not None else None
            if track is None:
                continue
            if skip_refuted and refuted(track):
                continue
            sim = cosine(getattr(track, "clip_ft", None), text_ft)
            if sim <= -1.0:
                continue
            if best is None or sim > best:
                best = sim
        if best is not None:
            scores[int(cid)] = float(best)
    return scores


class FeatureMemory:
    """The runtime side: one text feature per episode, features on the tracks.

    Holds the per-episode counters the arm is judged on. This benchmark's SR
    cannot resolve an effect smaller than about three trials, so a mechanism
    that cannot be SEEN in a counter cannot be evaluated at all -- an earlier
    campaign retired two correct hypotheses on nulls from knobs that were never
    connected.
    """

    COUNTERS = (
        "feature_crops_encoded",
        "feature_tracks_tagged",
        "feature_tracks_embedded",
        "feature_live_det_matches",
        "feature_admitted_live",
        "feature_admitted_prior",
        "feature_admitted_committed",
        "feature_local_picks",
        "feature_local_considered",
        "feature_local_committed",
        "feature_presence_changed_pick",
        "feature_local_too_large",
        "feature_prior_rounds",
        "feature_prior_argmax_changed",
        "same_class_fallback_applied",
        "stale_twins_retired_feature",
    )

    def __init__(self, encoder, cfg) -> None:
        self.encoder = encoder
        self.cfg = cfg
        self.target: str = ""
        self.text_ft: Optional[np.ndarray] = None
        self.counters: Dict[str, int] = {k: 0 for k in self.COUNTERS}
        # Prior crops are embedded once per (map file, track id): `apply_map`
        # rebuilds ObjectTrack objects from the snapshot on every episode, so a
        # cache keyed on the object would never hit.
        self._prior_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._admitted_seen: set = set()
        # One appearance commit per track per episode: without it a wrong pick
        # that survives its absence reading is re-picked at the next surface.
        self._picked: set = set()
        self.last_pick: Optional[Dict[str, Any]] = None
        # How a track's ground-plane position is read. The agent's object layer
        # resolves a linked component's centre, which a bare ellipsoid centre
        # does not; injected so this module never imports the layer.
        self._centre_of = lambda t: np.asarray(t.ellipsoid.center, dtype=float)[[0, 2]]

    def bind_centres(self, fn) -> None:
        self._centre_of = fn

    # ----------------------------------------------------------------- episode

    def set_target(self, target: str) -> None:
        self.target = str(target)
        prompt = str(self.cfg.prompt_template).format(target=self.target)
        self.text_ft = self.encoder.text_feature(prompt)
        self.counters = {k: 0 for k in self.COUNTERS}
        self._admitted_seen = set()
        self._picked = set()
        self.last_pick = None

    def sim(self, ft: Optional[np.ndarray]) -> float:
        return cosine(ft, self.text_ft)

    # ------------------------------------------------------------------ tracks

    def tag(self, track) -> float:
        """Record how much this track looks like the query.

        Only records. Whether a track may be committed to is decided by the
        argmax in `best_near`, never by this number crossing a bar -- the probe
        found no bar that admits the mug without admitting its neighbours, and
        DualMap does not use one.
        """
        sim = self.sim(getattr(track, "clip_ft", None))
        if track.feature_sim <= -1.0 and sim > -1.0:
            self.counters["feature_tracks_tagged"] += 1
        track.feature_sim = sim
        return sim

    def embed_detections(self, dets: Sequence[Any]) -> int:
        """One batched forward per keyframe over the admitted detections."""
        if not dets:
            return 0
        cap = int(self.cfg.max_live_crops_per_keyframe)
        pending = [d for d in dets if getattr(d, "clip_ft", None) is None
                   and getattr(d, "crop", None) is not None]
        if cap > 0:
            pending = pending[:cap]
        if not pending:
            return 0
        feats = self.encoder.encode_images([d.crop for d in pending])
        for det, ft in zip(pending, feats):
            det.clip_ft = ft
        self.counters["feature_crops_encoded"] += len(feats)
        return len(feats)

    def best_near(self, centre_xy, tracks, *, floor_key=None) -> Optional[Any]:
        """DualMap's local inquiry: the track near this surface that looks most
        like the query. Argmax, never a threshold, so it always answers.

        The candidate set is the live map only. A prior-map track the episode has
        not seen is a memory of another session, and DualMap's local map has no
        such thing -- it is built from this run's frames. Refuted tracks are out
        for the same reason a searched surface decays: the agent went and looked.
        """
        if self.text_ft is None:
            return None
        radius = float(self.cfg.local_radius_m)
        min_obs = int(self.cfg.local_min_obs)
        centre = np.asarray(centre_xy, dtype=float)
        eligible = []
        n_seen = 0
        for track in tracks:
            if not bool(getattr(track, "seen_live", False)):
                continue
            if int(getattr(track, "n_obs", 0)) < min_obs:
                continue
            if refuted(track) or int(track.id) in self._picked:
                continue
            if floor_key is not None and int(getattr(track, "floor_key", 0)) != int(floor_key):
                continue
            ft = getattr(track, "clip_ft", None)
            if ft is None:
                continue
            if not self._small_enough(track):
                self.counters["feature_local_too_large"] += 1
                continue
            xy = np.asarray(self._centre_of(track), dtype=float)
            if float(np.linalg.norm(xy - centre)) > radius:
                continue
            n_seen += 1
            eligible.append((track, self.sim(ft)))
        self.counters["feature_local_considered"] += n_seen
        if not eligible:
            return None
        sim_max = max(sim for _, sim in eligible)
        if bool(getattr(self.cfg, "fuse_presence", False)):
            beta = float(self.cfg.presence_beta)
            floor = float(self.cfg.presence_floor)
            scored = [
                (t, sim, feature_term(sim, sim_max, beta, floor) * _presence(t))
                for t, sim in eligible
            ]
        else:
            scored = [(t, sim, sim) for t, sim in eligible]
        best, best_sim, best_score = max(scored, key=lambda x: x[2])
        self._picked.add(int(best.id))
        self.counters["feature_local_picks"] += 1
        # Did the belief change the answer? The counter that decides whether the
        # fusion did anything at all, separately from whether SR moved.
        plain = max(eligible, key=lambda x: x[1])[0]
        if int(plain.id) != int(best.id):
            self.counters["feature_presence_changed_pick"] += 1
        self.last_pick = {
            "track_id": int(best.id), "label": str(best.label),
            "sim": round(float(best_sim), 4),
            "score": round(float(best_score), 4),
            "p": round(_presence(best), 4),
            "n_considered": n_seen,
            "argmax_sim_track": int(plain.id),
        }
        return best

    def _small_enough(self, track) -> bool:
        """Could this track be the object, on size alone?

        The ellipsoid's horizontal diameter against `local_max_extent_m`. A
        track wider than that is the furniture the object rests on, and a crop
        of a bed matches "a photo of a mug" better than forty pixels of the
        real one -- which is what the first version of this pick kept choosing.
        """
        cap = float(getattr(self.cfg, "local_max_extent_m", 0.0) or 0.0)
        if cap <= 0.0:
            return True
        ell = getattr(track, "ellipsoid", None)
        axes = getattr(ell, "axes", None)
        if axes is None:
            return True
        a = np.asarray(axes, dtype=float)
        if a.size < 3:
            return True
        return float(2.0 * max(a[0], a[2])) <= cap