"""The simulator's debug view, live on a robot: pushed to RViz through the bridge.

`DebugVideo` writes [YOLOE segmentation overlay | top-down costmap] per step
to an mp4. This draws the SAME frames with the SAME functions and, instead of
a file, hands them to the ROS bridge, which republishes them as images for
`scripts/ros2/osg.rviz`:

    /osg/rgb          the rotated frame the pipeline actually consumes
    /osg/detections   that frame with the detector's boxes, masks and labels
    /osg/debug        the two-panel picture the simulator's video shows
    /osg/scene_graph  the scene graph as markers, in Nav2's `map` frame: every
                      object track as an ellipsoid coloured by presence belief
                      (green believed, red disbelieved, grey blacklisted),
                      containers, rooms, the trajectory, the current path, the
                      chosen frontier and goal, and a status line -- the storey
                      the agent is on and the storey it WANTS, which is the
                      operator's cue to carry the robot. Storeys stack in z by
                      `floor.virtual_storey_m`, exactly as the pipeline holds
                      them, so the two floors of a map do not overprint.

As in `DebugVideo`, the detector is re-run here for visualisation only; the
object layer is untouched and pipeline behaviour is unchanged. Best effort by
construction: a bridge that cannot take the images (an old one, or a dropped
call) is logged once and the run goes on -- the view is a diagnostic and must
never cost a step.
"""
from __future__ import annotations

import logging

import numpy as np

from .visualize import overlay_segmentation, render_costmap_bgr

log = logging.getLogger(__name__)


def _jsonable(value):
    """json.dumps fallback: numpy scalars/arrays, enums, dataclasses, and
    anything else as its repr -- the log must never raise."""
    import dataclasses
    import enum

    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        v = value.item()
        return v if not isinstance(v, float) or np.isfinite(v) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


class DebugStream:
    """Same `write(frame, agent, target, detector)` contract as `DebugVideo`."""

    RGB_TOPIC = "/osg/rgb"
    DET_TOPIC = "/osg/detections"
    PANEL_TOPIC = "/osg/debug"
    GRAPH_TOPIC = "/osg/scene_graph"
    GRID_TOPIC = "/osg/costmap"     # the pipeline's own costmap, this storey, as an OccupancyGrid
    PANEL_H = 640  # the simulator video's height; the costmap panel is square-ish
    JUMP_M = 1.0   # a step is seconds at 0.15 m/s; more than this is not driving

    # The agent's own diagnostic logs, the ones the episode record carries
    # (eval/record.py). Each step the entries ADDED since the last step go into
    # stream.jsonl, so the record's "why" -- which frontier and why, which
    # candidate was rejected on what number, when a goal was committed or given
    # up -- is on disk while the robot is still moving, not when it stops.
    AGENT_LOGS = ("state_log", "frontier_select_log", "candidate_reject_log",
                  "goal_commit_log", "giveup_log", "approach_retarget_log",
                  "approach_bbox_log", "presence_events", "floor_log",
                  "close_look_log", "attempt_log")

    def __init__(self, env, log_path=None, on_step=None) -> None:
        import cv2

        from ..mapping.costmap import PLANE as _PLANE

        self._cv2 = cv2
        self._plane = list(_PLANE)
        self._env = env
        self._traj: list = []
        self._dead = False
        self._wanted_floor = None  # last storey the agent asked for, for the cue
        # One JSON line per step on disk, written as the run goes. The episode
        # record is only written when a run ENDS properly (Ctrl-C or completion);
        # a run killed any other way -- twice now -- left no record and the RViz
        # stream gone with it, so "did it see the ball, why did it spin" could
        # not be answered. This is the same status line and target-track
        # numbers RViz shows, kept.
        self._log = open(log_path, "a") if log_path else None
        self._seen: dict = {}      # log name -> entries already written
        self._diag_last = None     # last approach_diag written, to write it on change
        self._on_step = on_step    # called with the step after each write (map checkpoints)
        self._markers_broken = False
        self._grid_broken = False
        self._yaw_deg = None

    def _push(self, topic: str, image: np.ndarray, encoding: str) -> None:
        if self._dead:
            return
        try:
            self._env.transport.publish_image(topic, image, encoding)
        except Exception as exc:  # noqa: BLE001 -- a debug view never ends a run
            self._dead = True
            log.warning("debug stream off: the bridge refused %s (%s)", topic, exc)

    # ---------------------------------------------------------------- markers

    @staticmethod
    def _ros(p3) -> tuple:
        """Pipeline world (x, y-up, z) -> ROS map (x, y, z): the inverse of
        frames.T_HAB_ROS, hab = (ros_x, ros_z, -ros_y)."""
        p3 = np.asarray(p3, dtype=float)
        return (float(p3[0]), float(-p3[2]), float(p3[1]))

    def _ros_xy(self, xy, floor: int) -> tuple:
        """A ground-plane point on a storey, lifted the way the pipeline lifts it."""
        z = float(floor) * float(getattr(self._env.cfg.floor, "virtual_storey_m", 3.0))
        return (float(xy[0]), float(-xy[1]), z + 0.05)

    def scene_graph_markers(self, agent, target: str, step: int) -> list:
        """The scene graph as plain dicts for the bridge's publish_markers.
        Every accessor is optional: an agent that lacks one (AscentNav, a test
        double) simply contributes nothing for it."""
        from ..core.labels import normalize_label

        tgt = normalize_label(target)
        ms: list = []
        # Only this storey, when asked (ros2.viz_other_storeys=false): the
        # graph of the floor the agent left disappears at the switch.
        here = int(getattr(getattr(agent, "floors", None), "current_id", 0) or 0)
        only_here = not bool(getattr(getattr(self._env.cfg, "ros2", None), "viz_other_storeys", True))

        def on_this_storey(node, attr: str) -> bool:
            return not only_here or int(getattr(node, attr, here) or 0) == here

        # Objects: one ellipsoid + label per track, from the same layer save_map reads.
        layer = getattr(agent, "object_layer", None)
        tracks = layer.tracks(include_blacklisted=True) if layer is not None else []
        tracks = [t for t in tracks if on_this_storey(t, "floor_key")]
        for t in tracks:
            ell = getattr(t, "ellipsoid", None)
            centre = np.asarray(ell.center if ell is not None else layer.center_of(t), dtype=float)
            axes = np.asarray(ell.axes if ell is not None else (0.15, 0.15, 0.15), dtype=float)
            pres = getattr(t, "presence", None)
            p = float(getattr(pres, "p", 1.0)) if pres is not None else 1.0
            is_t = normalize_label(str(t.label)) == tgt
            if bool(getattr(t, "blacklisted", False)):
                rgba = (0.5, 0.5, 0.5, 0.25)
            else:  # presence belief: green when believed, red when disbelieved
                rgba = (1.0 - p, p, 0.1, 0.55 if not is_t else 0.85)
            x, y, z = self._ros(centre)
            sx, sy, sz = (float(2 * a) for a in (axes[0], axes[2], axes[1]))  # (x, z, y-up) -> (x, y, z)
            ms.append({"ns": "objects", "id": int(t.id), "type": "sphere",
                       "xyz": (x, y, z), "scale": (max(sx, 0.05), max(sy, 0.05), max(sz, 0.05)),
                       "rgba": rgba})
            text = f"{t.label} {p:.2f}"
            if is_t:
                # The target's label carries the numbers the candidate gate
                # reads (object_layer.candidates), so "it saw the ball and
                # kept exploring" can be answered from RViz rather than after
                # the run: detection score, box area, observations, evidence,
                # and whether it is still proposal-only.
                text += (f"  s={float(getattr(t, 'best_score', 0.0)):.2f}"
                         f" box={float(getattr(t, 'best_bbox_px', 0.0)):.0f}px"
                         f" n={int(getattr(t, 'n_obs', 0))}"
                         f" ev={float(getattr(t, 'evidence', 0.0)):.2f}"
                         + (f" PROPOSAL-ONLY n_prop={int(getattr(t, 'n_proposal_obs', 0))} sim={float(getattr(t, 'proposal_sim', 0.0) or 0.0):.2f}"
                            if bool(getattr(t, 'proposal_only', False)) else "")
                         + (" BLACKLISTED" if bool(getattr(t, 'blacklisted', False)) else ""))
            ms.append({"ns": "labels", "id": int(t.id), "type": "text",
                       "xyz": (x, y, z + sz / 2 + 0.15), "scale": (0.0, 0.0, 0.12),
                       "rgba": (1.0, 0.2, 0.2, 1.0) if is_t else (1.0, 1.0, 1.0, 0.9),
                       "text": text})
        graph = getattr(agent, "scene_graph", None)
        for c in (getattr(graph, "containers", {}) or {}).values():
            if not on_this_storey(c, "floor"):
                continue
            x, y, z = self._ros(c.center)
            ms.append({"ns": "containers", "id": int(c.id), "type": "cube",
                       "xyz": (x, y, z), "scale": (0.6, 0.6, 0.06),
                       "rgba": (0.2, 0.6, 1.0, 0.25)})
            ms.append({"ns": "container_labels", "id": int(c.id), "type": "text",
                       "xyz": (x, y, z + 0.25), "scale": (0.0, 0.0, 0.1),
                       "rgba": (0.5, 0.8, 1.0, 0.9), "text": str(c.label)})
        for r in (getattr(graph, "rooms", {}) or {}).values():
            if not on_this_storey(r, "floor"):
                continue
            x, y, z = self._ros_xy(r.centroid_xy, int(getattr(r, "floor", 0)))
            ms.append({"ns": "rooms", "id": int(r.id), "type": "text",
                       "xyz": (x, y, z + 1.6), "scale": (0.0, 0.0, 0.25),
                       "rgba": (1.0, 0.85, 0.3, 0.9), "text": str(r.label or f"room {r.id}")})
        # Where it is going: trajectory, planned path, chosen frontier, goal.
        floors = getattr(agent, "floors", None)
        floor = int(getattr(floors, "current_id", 0)) if floors is not None else 0
        if len(self._traj) >= 2:
            ms.append({"ns": "trajectory", "id": 0, "type": "line_strip",
                       "scale": (0.03, 0.0, 0.0), "rgba": (1.0, 0.6, 0.0, 0.9),
                       "points": [self._ros_xy(p_, floor) for p_ in self._traj]})
        path = getattr(agent, "_current_path", None)
        if path is not None and len(path) >= 2:
            ms.append({"ns": "path", "id": 0, "type": "line_strip",
                       "scale": (0.04, 0.0, 0.0), "rgba": (0.1, 1.0, 0.1, 0.9),
                       "points": [self._ros_xy(p_, floor) for p_ in np.asarray(path)]})
        fr = getattr(agent, "_current_frontier", None)
        if fr is not None and getattr(fr, "centroid_xy", None) is not None:
            x, y, z = self._ros_xy(fr.centroid_xy, floor)
            ms.append({"ns": "frontier", "id": 0, "type": "cylinder", "xyz": (x, y, z),
                       "scale": (0.5, 0.5, 0.05), "rgba": (1.0, 0.0, 1.0, 0.6)})
        goal = getattr(agent, "_goal_xy", None)
        if goal is not None:
            x, y, z = self._ros_xy(goal, floor)
            ms.append({"ns": "goal", "id": 0, "type": "sphere", "xyz": (x, y, z + 0.15),
                       "scale": (0.3, 0.3, 0.3), "rgba": (1.0, 0.1, 0.0, 0.9)})
        # The status line, and the operator's cue.
        want = getattr(getattr(agent, "exploration", None), "requested_floor", None)
        state = getattr(getattr(agent, "state", None), "name", str(getattr(agent, "state", "")))
        status = f"{target}  step {step}  floor {floor}  {state}"
        if want is not None and int(want) != floor:
            status += f"   >>> WANTS FLOOR {int(want)}: carry the robot, then  ros2 topic pub --once /osg/floor std_msgs/Int32 '{{data: {int(want)}}}'"
            stairs = (getattr(agent, "stats", None) or {}).get("stairs_wait_goal_ros")
            if stairs is not None:
                where = "waiting at" if state == "EXPLORE" else "driving to"
                status += f"   [{where} the stairs ({stairs[0]:.2f}, {stairs[1]:.2f})]"
        ax, ay, az = self._ros_xy(self._traj[-1], floor) if self._traj else (0.0, 0.0, 0.0)
        ms.append({"ns": "status", "id": 0, "type": "text", "xyz": (ax, ay, az + 2.2),
                   "scale": (0.0, 0.0, 0.3), "rgba": (1.0, 1.0, 1.0, 1.0), "text": status})
        if self._log is not None:
            import json, time as _time
            rec = {"t": round(_time.time(), 2), "step": int(step), "state": state, "floor": floor,
                   "yaw_deg": self._yaw_deg,
                   "wants_floor": None if want is None else int(want),
                   "agent_xy": [round(float(v), 3) for v in self._traj[-1]] if self._traj else None,
                   "goal_xy": None if goal is None else [round(float(v), 3) for v in goal],
                   "frontier_xy": None if fr is None or getattr(fr, "centroid_xy", None) is None
                   else [round(float(v), 3) for v in fr.centroid_xy],
                   "n_tracks": len(tracks),
                   # startswith, not the first word: "sports ball" is two words, and
                   # a split()[0] filter hid a 0.71 ball track for a whole run.
                   "target_tracks": [m["text"] for m in ms if m["ns"] == "labels" and normalize_label(m["text"]).startswith(tgt + " ")]}
            rec.update(self._agent_log_deltas(agent))
            self._log.write(json.dumps(rec, default=_jsonable) + "\n"); self._log.flush()
        if want is not None and int(want) != floor and want != self._wanted_floor:
            print(f"[robot] the agent wants floor {int(want)} (it is on floor {floor}): "
                  f"carry it there, then publish  /osg/floor {int(want)}", flush=True)
        self._wanted_floor = want if (want is not None and int(want) != floor) else None
        return ms

    def _agent_log_deltas(self, agent) -> dict:
        """The entries each agent log gained since the last step, plus
        approach_diag whenever it changed. Missing attributes are simply
        absent -- the AscentNav agent has few of these."""
        out: dict = {}
        for name in self.AGENT_LOGS:
            log = getattr(agent, name, None)
            if not isinstance(log, (list, tuple)):
                continue
            seen = self._seen.get(name, 0)
            if len(log) > seen:
                out[name] = list(log[seen:])
            self._seen[name] = len(log)
        try:
            diag = getattr(agent, "approach_diag", None)
        except Exception:   # a property that computes; never let it cost a step
            diag = None
        if diag and diag != self._diag_last:
            out["approach_diag"] = diag
            self._diag_last = diag
        reason = getattr(agent, "approach_stop_reason", None)
        if reason is not None and reason != self._seen.get("approach_stop_reason"):
            out["approach_stop_reason"] = reason
            self._seen["approach_stop_reason"] = reason
        return out

    def _push_markers(self, agent, target: str, step: int) -> None:
        """Build AND send inside the guard: the picture is assembled from the
        agent's internals by duck typing, and a missing attribute on some
        agent must switch the view off, not end the run."""
        # Build first, outside the bridge guard: building is also what writes
        # the stream.jsonl line, and that line must keep coming after the bridge
        # has stopped taking pictures.
        if self._markers_broken:
            return
        try:
            markers = self.scene_graph_markers(agent, target, step)
        except Exception as exc:  # noqa: BLE001
            self._markers_broken = True
            log.warning("scene-graph view off: building it failed (%r)", exc)
            return
        if self._dead:
            return
        try:
            self._env.transport.publish_markers(self.GRAPH_TOPIC, markers, "map")
        except Exception as exc:  # noqa: BLE001
            self._dead = True
            log.warning("debug stream off: %s failed (%r)", self.GRAPH_TOPIC, exc)

    def write(self, frame, agent, target: str, detector) -> None:
        cv2 = self._cv2
        self._note_pose(frame)
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        self._push(self.RGB_TOPIC, rgb, "rgb8")

        if hasattr(agent, "debug_panel"):  # AscentNav draws its own maps
            self._push(self.PANEL_TOPIC, agent.debug_panel(frame), "bgr8")
            self._traj.append(frame.camera_position[self._plane])
            self._push_markers(agent, target, len(self._traj))
            self._after_step(len(self._traj))
            return

        dets = detector.detect(rgb)  # viz-only; does not update the object layer
        seg = overlay_segmentation(rgb, dets, target)
        self._push(self.DET_TOPIC, seg, "bgr8")

        agent_xy = frame.camera_position[self._plane]
        self._traj.append(agent_xy)
        h = self.PANEL_H
        seg_small = cv2.resize(seg, (int(round(seg.shape[1] * h / seg.shape[0])), h))
        cm = render_costmap_bgr(
            agent.costmap, agent_xy, self._traj,
            path_xy=getattr(agent, "_current_path", None),
            chosen_frontier=getattr(agent, "_current_frontier", None),
            out_h=h,
        )
        cm = cv2.resize(cm, (int(round(cm.shape[1] * h / cm.shape[0])), h))
        panel = cv2.hconcat([seg_small, cm])
        cv2.putText(panel, f"{target}  step {len(self._traj)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        self._push(self.PANEL_TOPIC, panel, "bgr8")
        self._push_markers(agent, target, len(self._traj))
        self._push_grid(agent)
        self._after_step(len(self._traj))

    def _push_grid(self, agent) -> None:
        """This storey's costmap -- what the frontiers are chosen on -- beside
        the robot's own map in the map window. Same guard as the markers."""
        if self._dead or self._grid_broken:
            return
        cm = getattr(agent, "costmap", None)
        grid = getattr(cm, "grid", None)
        if grid is None:
            return
        try:
            floor = int(getattr(getattr(agent, "floors", None), "current_id", 0) or 0)
            z = floor * float(self._env.cfg.floor.virtual_storey_m)
            self._env.transport.publish_grid(self.GRID_TOPIC, grid, cm.origin, cm.resolution, "map", z)
        except Exception as exc:  # noqa: BLE001
            self._grid_broken = True
            log.warning("costmap view off: %s failed (%r)", self.GRID_TOPIC, exc)

    def _note_pose(self, frame) -> None:
        """The camera's heading for the log, and a loud line when the pose
        jumps. On the robot the pose is Nav2's localisation, and a run whose
        AMCL had not converged carried a 6 m jump in one 2.5 s step: every
        object placed before it was somewhere else, one on the opposite wall.
        Nothing in the pipeline can tell that from motion, but a person
        watching the log can, if it is said."""
        try:
            T = np.asarray(frame.T_wc, dtype=float)
            fwd = T[:3, 2]                                   # optical axis, pipeline world
            self._yaw_deg = round(float(np.degrees(np.arctan2(-fwd[2], fwd[0]))), 1)  # ROS yaw
            xy = T[[0, 2], 3]
            if self._traj:
                d = float(np.linalg.norm(xy - np.asarray(self._traj[-1], dtype=float)))
                if d > self.JUMP_M:
                    print(f"[robot] POSE JUMP: {d:.2f} m in one step (to ROS "
                          f"({xy[0]:.2f}, {-xy[1]:.2f})). Localisation moved, not the robot: "
                          "everything mapped before this step is misplaced.", flush=True)
        except Exception:  # noqa: BLE001
            self._yaw_deg = None

    def _after_step(self, step: int) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(step)
        except Exception as exc:  # noqa: BLE001
            log.warning("on_step hook failed at step %d (%r)", step, exc)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
