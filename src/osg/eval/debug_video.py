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
        self._path = out_dir / "viz" / "debug" / f"{tag}.mp4"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Opened lazily: an agent that renders its own panel decides the frame
        # size, and only it knows what that is.
        self._vw = None

    def _writer(self, size):
        if self._vw is None:
            self._vw = self._cv2.VideoWriter(
                str(self._path), self._cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
        return self._vw

    def write(self, frame, agent, target: str, detector) -> None:
        cv2 = self._cv2
        # An agent that draws its own maps renders itself -- `AscentNavAgent`
        # holds ASCENT's obstacle/value maps, which the costmap path below
        # cannot show.
        if hasattr(agent, "debug_panel"):
            panel = agent.debug_panel(frame)
            self._writer((panel.shape[1], panel.shape[0])).write(panel)
            return
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
        self._writer((self._w, self._h)).write(panel)

    def close(self) -> None:
        if self._vw is not None:
            self._vw.release()
