"""The protocol between the pipeline (py3.9) and the ROS bridge (py3.10).

`multiprocessing.connection` rather than HTTP or pyzmq: it is in the standard
library of both interpreters, so neither environment grows a dependency, and a
request/reply socket is all this needs. One client at a time, one round trip
per call, no queue.

Arrays are encoded explicitly as `{dtype, shape, bytes}` instead of being
pickled. A pickled ndarray carries the writing numpy's internal reconstruction
path, and the two sides here are different numpy builds (1.23 pinned in conda,
whatever apt ships on 3.10); the explicit form is also what makes a frame
payload readable in a traceback when something disagrees about a shape.

Frame payloads travel in ROS coordinates and ROS units -- raw depth, the camera
matrix as published, `T_map_cam` as TF reports it. Every conversion into the
pipeline's world lives in `frames.py`, on the OSG side, where it is unit-tested
without a ROS installation present. The bridge converts nothing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# The ops a bridge answers. Kept here so both sides import the same spelling:
# a typo'd op name should fail at the dispatch table, not silently no-op.
OPS = (
    "ping",             # -> {"node": str, "frames": int, "ros_time": float}
    "get_frame",        # (after_seq, timeout_s) -> frame payload, or None on timeout
    "send_goal",        # (x, y, yaw, frame_id) -> {"goal_id": int}
    "nav_status",       # -> {"state": ..., "goal_id": int, "distance_remaining": float}
    "cancel",           # -> {"cancelled": bool}
    "execute",          # (action, forward_m, turn_deg) -> {"achieved": float, "timed_out": bool}
    "look",             # (tilt_delta_deg) -> {"tilt_deg": float}
    "pop_floor_switch", # -> {"floor": Optional[int]}
    "last_goal",        # -> the goal dict handed to Nav2, or None
    "publish_image",    # (topic, image, encoding) -> {"published": True}; RViz only, /osg/* only
    "publish_markers",  # (topic, markers, frame_id) -> {"published": int}; RViz only, /osg/* only
    "publish_grid",     # (topic, grid, origin_xy, resolution, frame_id, z) -> {"published": True}; RViz only, /osg/* only
    "shutdown",         # -> {"bye": True}
)

# What `nav_status()["state"]` may be. `active` is the only one that means the
# robot is still being driven; `aborted` and `rejected` both mean Nav2 will not
# reach this goal, which is the pipeline's "blocked".
NAV_STATES = ("idle", "active", "succeeded", "aborted", "rejected", "canceled")

_ND = "__nd__"


class RemoteError(RuntimeError):
    """The bridge raised while handling an op. Carries its traceback text."""

    def __init__(self, op: str, message: str, remote_traceback: str = "") -> None:
        super().__init__(f"{op}: {message}")
        self.op = op
        self.remote_traceback = remote_traceback


def encode_array(a: np.ndarray) -> dict:
    # Shape is read BEFORE `ascontiguousarray`, which promotes a 0-d array to
    # shape (1,) and would quietly change what the other side receives.
    shape = list(a.shape)
    return {_ND: 1, "dtype": a.dtype.str, "shape": shape,
            "data": np.ascontiguousarray(a).tobytes()}


def decode_array(d: dict) -> np.ndarray:
    a = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return a.reshape(tuple(d["shape"])).copy()  # copy: frombuffer is read-only


def encode(obj: Any) -> Any:
    """Recursively replace ndarrays with their explicit form."""
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if isinstance(obj, np.generic):  # np.float32(...) etc: pickles fine, but be explicit
        return obj.item()
    return obj


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get(_ND):
            return decode_array(obj)
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def call(conn, op: str, **args) -> Any:
    """One round trip. Raises `RemoteError` if the bridge reported a failure."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    conn.send({"op": op, "args": encode(args)})
    reply = conn.recv()
    if not reply.get("ok", False):
        raise RemoteError(op, str(reply.get("error", "")), str(reply.get("traceback", "")))
    return decode(reply.get("result"))


def serve_once(conn, handlers: dict) -> bool:
    """Read one request, dispatch it, write the reply.

    Returns False when the client asked the bridge to shut down or hung up, so
    the accept loop can decide whether to wait for another client. Exceptions
    become replies rather than killing the bridge: a pipeline that asks for a
    frame before the camera has published should get an error it can retry, not
    a dead socket.
    """
    import traceback

    try:
        request = conn.recv()
    except EOFError:
        return False
    op = str(request.get("op", ""))
    args = decode(request.get("args") or {})
    if op == "shutdown":
        conn.send({"ok": True, "result": {"bye": True}})
        return False
    try:
        handler = handlers[op]
    except KeyError:
        conn.send({"ok": False, "error": f"unknown op {op!r}", "traceback": ""})
        return True
    try:
        result = handler(**args)
    except Exception as exc:  # noqa: BLE001 -- the reply IS the error channel
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()})
        return True
    conn.send({"ok": True, "result": encode(result)})
    return True


def parse_addr(addr: str):
    """`"host:port"` -> the tuple `multiprocessing.connection` wants."""
    host, _, port = str(addr).rpartition(":")
    if not host or not port:
        raise ValueError(f"bridge address must be host:port, got {addr!r}")
    return (host, int(port))
