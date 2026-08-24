"""Per-episode debug video, for eyeballing what the agent actually saw."""
from __future__ import annotations

from pathlib import Path

from .visualize import overlay_segmentation, render_costmap_bgr


class DebugVideo:
    """Per-episode debug video: each frame is [live RGB + YOLOE segmentation
    overlay | top-down costmap] at every step. The detector is re-run here for
    visualization only (it does not feed the object layer), so pipeline
    behaviour / SR is unchanged. Enabled by eval.debug_frames."""

    def __init__(self, cfg, out_dir: Path, tag: str) -> None:
        import cv2

        from ..mapping.costmap import PLANE as _PLANE

        self._cv2 = cv2
        self._plane = list(_PLANE)
        self._cm_w = 480
        self._h = cfg.eval.rgb_height
        self._w = cfg.eval.rgb_width + self._cm_w
        self._traj: list = []
        path = out_dir / "viz" / "debug" / f"{tag}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._vw = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (self._w, self._h)
        )

    def write(self, frame, agent, target: str, detector) -> None:
        cv2 = self._cv2
        agent_xy = frame.camera_position[self._plane]
        self._traj.append(agent_xy)
        dets = detector.detect(frame.rgb)  # viz-only; does not update object layer
        seg = overlay_segmentation(frame.rgb, dets, target)
        seg = cv2.resize(seg, (self._w - self._cm_w, self._h))
        cm = render_costmap_bgr(
            agent.costmap, agent_xy, self._traj,
            path_xy=getattr(agent, "_current_path", None),
            chosen_frontier=getattr(agent, "_current_frontier", None),
            out_h=self._h,
        )
        cm = cv2.resize(cm, (self._cm_w, self._h))
        panel = cv2.hconcat([seg, cm])
        cv2.putText(panel, f"{target}  step {len(self._traj)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        self._vw.write(panel)

    def close(self) -> None:
        self._vw.release()
