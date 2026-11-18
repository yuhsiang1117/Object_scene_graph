"""The pipeline and the ROS bridge must agree over the socket.

They are different interpreters with different numpy builds, so the codec is
explicit rather than pickled, and these tests run it over a real
`multiprocessing.connection` pair -- the mistakes worth catching (a read-only
`frombuffer` view, a dtype that survives locally but not across) do not show up
against an in-process dict.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from osg.ros2 import wire


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64", "bool", "int64"])
def test_every_dtype_the_bridge_sends_round_trips(dtype):
    a = (np.arange(24).reshape(2, 3, 4) % 7).astype(dtype)
    out = wire.decode(wire.encode(a))
    assert out.dtype == a.dtype and out.shape == a.shape
    assert np.array_equal(out, a)


def test_a_decoded_array_is_writable():
    """`np.frombuffer` returns a read-only view. The costmap writes into the
    depth it is given, so a view would fail deep inside mapping instead of
    here."""
    out = wire.decode(wire.encode(np.zeros((2, 2), dtype=np.float32)))
    out[0, 0] = 1.0  # must not raise


def test_nested_payloads_and_numpy_scalars():
    payload = {"rgb": np.zeros((2, 2, 3), np.uint8), "K": np.eye(3),
               "meta": {"stamp": np.float64(1.5), "shape": [np.int64(4), 5]},
               "frame_id": "camera_optical", "seq": 7}
    out = wire.decode(wire.encode(payload))
    assert out["frame_id"] == "camera_optical" and out["seq"] == 7
    assert out["meta"]["stamp"] == 1.5 and out["meta"]["shape"] == [4, 5]
    assert out["rgb"].shape == (2, 2, 3) and np.allclose(out["K"], np.eye(3))


def test_empty_and_zero_dimensional_arrays_survive():
    for a in (np.zeros((0, 4), np.float32), np.array(3.5)):
        out = wire.decode(wire.encode(a))
        assert out.shape == a.shape and np.array_equal(out, a)


def test_parse_addr_rejects_a_bare_host():
    assert wire.parse_addr("127.0.0.1:18765") == ("127.0.0.1", 18765)
    with pytest.raises(ValueError, match="host:port"):
        wire.parse_addr("127.0.0.1")


def test_an_unknown_op_is_refused_before_it_is_sent():
    with pytest.raises(ValueError, match="unknown op"):
        wire.call(None, "drive_into_the_wall")


def _serve(listener, handlers, rounds=8):
    conn = listener.accept()
    for _ in range(rounds):
        if not wire.serve_once(conn, handlers):
            break
    conn.close()


def _pair(handlers, rounds=8):
    """A bridge-shaped server on a thread, and a client connected to it."""
    from multiprocessing.connection import Client, Listener

    listener = Listener(("127.0.0.1", 0), authkey=b"test")
    thread = threading.Thread(target=_serve, args=(listener, handlers, rounds), daemon=True)
    thread.start()
    client = Client(listener.address, authkey=b"test")
    return client, listener, thread


def test_a_call_round_trips_over_a_real_socket():
    client, listener, thread = _pair({
        "ping": lambda: {"node": "bridge", "frames": 3},
        "get_frame": lambda min_stamp, timeout_s: {
            "stamp": min_stamp + 1.0, "depth": np.full((2, 2), 1.5, np.float32)},
    })
    try:
        assert wire.call(client, "ping")["node"] == "bridge"
        frame = wire.call(client, "get_frame", min_stamp=10.0, timeout_s=1.0)
        assert frame["stamp"] == 11.0
        assert np.allclose(frame["depth"], 1.5)
    finally:
        client.close(); listener.close(); thread.join(timeout=2)


def test_a_raising_handler_becomes_an_error_reply_not_a_dead_socket():
    """The bridge serves one client for a whole run. An op that raises -- a TF
    lookup that is not ready yet -- must not take the connection with it."""
    def boom():
        raise ValueError("tf lookup failed")

    client, listener, thread = _pair({"cancel": boom, "ping": lambda: {"node": "bridge"}})
    try:
        with pytest.raises(wire.RemoteError) as excinfo:
            wire.call(client, "cancel")
        assert "tf lookup failed" in str(excinfo.value)
        assert "ValueError" in excinfo.value.remote_traceback
        assert wire.call(client, "ping")["node"] == "bridge"  # still usable
    finally:
        client.close(); listener.close(); thread.join(timeout=2)


def test_shutdown_ends_the_session():
    client, listener, thread = _pair({"ping": lambda: {}})
    try:
        assert wire.call(client, "shutdown") == {"bye": True}
        thread.join(timeout=2)
        assert not thread.is_alive()
    finally:
        client.close(); listener.close()
