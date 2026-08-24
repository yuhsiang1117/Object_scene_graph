"""Ground truth, kept strictly on the runner's side of the wall.

The most valuable engineering output of the dynamic-scene work is not a
mechanism, it is an instrument. Twice, a conclusion was drawn about WHY episodes
failed, acted on, and turned out to have been a guess -- because
`episodes.jsonl` could not separate "the agent never pointed a camera at the new
location" from "it did, and the detection fell under the gate". Those have
entirely different fixes and nothing distinguished them.

`GroundTruthVisibility` projects the authored target position into every frame,
rejects it if it is outside the image or behind the camera, and rejects it again
if the depth buffer says something solid is in front. What survives is "the
object was in view, unoccluded, at this range" -- and on keyframes, whether the
detector then named it.

This is GROUND TRUTH. The agent is never given it, never sees these fields, and
nothing here writes to the agent. That contract is the whole reason the file
exists separately from the agent it measures.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np


class GroundTruthVisibility:
    """Did the agent ever LOOK at where the object actually is?

    The dominant failure of the dynamic benchmark is that 57 of 96 episodes
    never perceive the object at its new pose, and 48 of them still hold it at
    the old one. That has two entirely different causes with two entirely
    different fixes -- the agent never pointed a camera at the new location, or
    it did and the detection fell under the gate -- and nothing in episodes.jsonl
    separates them. Everything downstream of that split has been guessed at
    twice now.

    So: project the authored target position into every frame, reject it if it
    is outside the image or behind the camera, and reject it again if the depth
    buffer says something solid is in front of it. What survives is "the object
    was in view, unoccluded, at this range".

    This is GROUND TRUTH and lives in the runner. The agent is never given it,
    never sees these fields, and nothing here writes to the agent.
    """

    # The projected point is the object's CENTRE; depth returns its front face,
    # which for a cracker box is ~7 cm nearer. Anything closer than this is a
    # different surface in the way.
    OCCLUSION_TOL_M = 0.25

    def __init__(self, target_xyz: Optional[Sequence[float]]) -> None:
        self.target = None if target_xyz is None else np.asarray(target_xyz, dtype=float)
        self.frames = 0
        self.in_view = 0
        self.min_range_m = float("inf")
        self.close_frames = 0  # in view within 3 m, where detection is plausible
        self.kf_in_view = 0
        self.kf_detected = 0
        self.best_offaxis = float("inf")
        self.best_det_score = 0.0
        self.visible_fraction_sum = 0.0
        self.by_framing = {k: [0, 0] for k in
                           ("close_centred", "close_peripheral",
                            "far_centred", "far_peripheral")}

    def observe(self, frame) -> None:
        if self.target is None:
            return
        self.frames += 1
        seen = self._project(frame)
        if seen is None:
            return
        z = seen[2]
        self.in_view += 1
        self.min_range_m = min(self.min_range_m, z)
        if z <= 3.0:
            self.close_frames += 1

    # -------------------------------------------------------- keyframe half
    #
    # The step-by-step counters above answer "did the agent look at it". They
    # cannot answer "and did the detector see it", because detection only runs
    # on keyframes -- so a keyframe is the correct denominator for recall, and
    # the only place an in-situ miss can be attributed.
    #
    # Condition H tested the hypothesis that the 30 looked-and-missed episodes
    # were detections sitting just under the admission gate, by lowering the
    # gate for the four classes a 900-pose census called cheap. The population
    # moved by one episode. The census had measured recall at AUTHORED
    # viewpoints -- rings around the object, pointed at it -- while the agent
    # arrives at a median 1.07 m on whatever heading the follower left it with.
    # What is missing is the framing: an object clipped to the edge of a
    # wide-FOV frame is indistinguishable, in the counters above, from one
    # centred at the same range.

    def observe_keyframe(self, frame, dets, target_label: str) -> None:
        """Was the object in frame, and did the detector call it by name?"""
        if self.target is None:
            return
        seen = self._project(frame)
        if seen is None:
            return
        u, v, z, fraction = seen
        self.visible_fraction_sum += fraction
        k = frame.intrinsics
        # 0 at the optical axis, 1 at the nearer image edge, more into a corner.
        offaxis = math.hypot((u - k.cx) / (0.5 * k.width), (v - k.cy) / (0.5 * k.height))
        self.kf_in_view += 1
        self.best_offaxis = min(self.best_offaxis, offaxis)
        want = str(target_label).lower().replace("_", " ").strip()
        best = 0.0
        for det in dets or []:
            if str(det.label).lower().replace("_", " ").strip() != want:
                continue
            x1, y1, x2, y2 = [float(c) for c in det.bbox_xyxy]
            if x1 - 8.0 <= u <= x2 + 8.0 and y1 - 8.0 <= v <= y2 + 8.0:
                best = max(best, float(det.score))
        if best > 0.0:
            self.kf_detected += 1
            self.best_det_score = max(self.best_det_score, best)
        # Recall conditioned on framing, which is the thing H could not see.
        centred = offaxis <= 0.6
        if z <= 3.0:
            key = "close_centred" if centred else "close_peripheral"
        else:
            key = "far_centred" if centred else "far_peripheral"
        self.by_framing[key][0] += 1
        self.by_framing[key][1] += int(best > 0.0)

    # Seven points, not one: the centre and +-6 cm on each axis, which is inside
    # a YCB object rather than around it. Testing the centre pixel alone is far
    # too permissive -- an object nine tenths hidden behind a chair back, with
    # only its middle showing, passes -- and an instrument that counts those as
    # "the agent looked at it" would understate in-situ recall by exactly the
    # frames where the detector had no chance. The probe this is compared
    # against requires a real pixel count and then keeps the ten best views, so
    # the comparison is only honest if this end is not counting slivers.
    PROBE_OFFSETS_M = 0.06
    MIN_VISIBLE_FRACTION = 0.5

    def _project(self, frame):
        """(u, v, z, visible fraction) at the object's centre, or None."""
        k = frame.intrinsics
        r = self.PROBE_OFFSETS_M
        samples = [self.target]
        for axis in range(3):
            for sign in (-1.0, 1.0):
                q = self.target.copy()
                q[axis] += sign * r
                samples.append(q)
        centre = None
        visible = 0
        for i, point in enumerate(samples):
            p_c = frame.T_cw[:3, :3] @ point + frame.T_cw[:3, 3]
            z = float(p_c[2])
            if z <= 1e-3:
                continue
            u = k.fx * p_c[0] / z + k.cx
            v = k.fy * p_c[1] / z + k.cy
            if not (0 <= u < k.width and 0 <= v < k.height):
                continue
            d = float(frame.depth[int(v), int(u)])
            if d > 1e-3 and d < z - self.OCCLUSION_TOL_M:
                continue
            visible += 1
            if i == 0:
                centre = (u, v, z)
        if centre is None:
            return None
        fraction = visible / len(samples)
        if fraction < self.MIN_VISIBLE_FRACTION:
            return None
        return centre[0], centre[1], centre[2], fraction

    def fields(self) -> Dict[str, Any]:
        out = {
            "gt_frames": self.frames,
            "gt_in_view_frames": self.in_view,
            "gt_in_view_close_frames": self.close_frames,
            "gt_min_range_m": (round(self.min_range_m, 3)
                               if self.min_range_m < float("inf") else None),
            "gt_kf_in_view": self.kf_in_view,
            "gt_kf_detected": self.kf_detected,
            "gt_best_offaxis": (round(self.best_offaxis, 3)
                                if self.best_offaxis < float("inf") else None),
            "gt_best_det_score": round(self.best_det_score, 3),
            "gt_mean_visible_fraction": (round(self.visible_fraction_sum / self.kf_in_view, 3)
                                         if self.kf_in_view else None),
        }
        for key, (n, hit) in self.by_framing.items():
            out[f"gt_kf_{key}"] = n
            out[f"gt_kf_{key}_detected"] = hit
        return out
