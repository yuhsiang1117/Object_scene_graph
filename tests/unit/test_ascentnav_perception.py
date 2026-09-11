"""The served-model clients, the fail-loud contract, per-step tagging, and the
depth the maps see."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from osg.perception.ascent_models import (
    GroundingDinoStairDetector,
    PerceptionUnavailable,
    RamTagger,
    probe_perception_servers,
)


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status = body, status

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, handler):
    """Route urlopen by URL path to a handler returning a JSON-able reply."""
    sent = []

    def fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        payload = json.loads(req.data) if getattr(req, "data", None) else None
        sent.append((url, payload))
        out = handler(url, payload)
        if isinstance(out, Exception):
            raise out
        return _Resp(json.dumps(out).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return sent


def _mask_b64(h, w, box):
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return base64.b64encode(m.tobytes()).decode()


def test_gdino_keeps_only_stair_boxes_above_the_bar_and_unions_their_masks(monkeypatch):
    h, w = 48, 64

    def handler(url, payload):
        if url.endswith("/gdino"):
            assert payload["caption"] == "stair ."
            return {"boxes": [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0]],
                    "logits": [0.9, 0.7, 0.3], "phrases": ["stair", "stair", "stair"]}
        bx = payload["bbox"]
        return {"cropped_mask": _mask_b64(h, w, bx)}

    _serve(monkeypatch, handler)
    d = GroundingDinoStairDetector(strict=True)
    rgb = np.zeros((h, w, 3), np.uint8)
    assert len(d.boxes(rgb)) == 2
    m = d.mask(rgb)
    assert m.shape == (h, w) and m[:24, :32].all() and m[24:, 32:].all() and not m[24:, :32].any()


def test_gdino_ignores_other_phrases(monkeypatch):
    _serve(monkeypatch, lambda url, p: {"boxes": [[0, 0, 1, 1]], "logits": [0.99], "phrases": ["door"]})
    assert GroundingDinoStairDetector().boxes(np.zeros((8, 8, 3), np.uint8)) == []


def test_ram_splits_the_tag_string(monkeypatch):
    _serve(monkeypatch, lambda url, p: ["armchair | lamp | living room", "扶手椅"])
    assert RamTagger().tags(np.zeros((8, 8, 3), np.uint8)) == ["armchair", "lamp", "living room"]


def test_strict_clients_raise_and_lenient_ones_return_empty(monkeypatch):
    _serve(monkeypatch, lambda url, p: OSError("connection refused"))
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(PerceptionUnavailable):
        RamTagger(strict=True).tags(rgb)
    with pytest.raises(PerceptionUnavailable):
        GroundingDinoStairDetector(strict=True).boxes(rgb)
    assert RamTagger(strict=False).tags(rgb) == []
    assert GroundingDinoStairDetector(strict=False).mask(rgb) is None


def test_a_strict_blip2_raises_instead_of_scoring_zero(monkeypatch):
    """The single most dangerous silent failure: a cosine of 0 from a dead
    server means ASCENT's gate never latches and the agent never STOPs."""
    from osg.perception.image_text import Blip2ItmScorer

    _serve(monkeypatch, lambda url, p: OSError("refused"))
    with pytest.raises(PerceptionUnavailable):
        Blip2ItmScorer(strict=True).score(np.zeros((8, 8, 3), np.uint8), ["a bed"])


def test_a_strict_dfine_raises(monkeypatch):
    from osg.perception.detector import DFineDetector

    _serve(monkeypatch, lambda url, p: OSError("refused"))
    with pytest.raises(PerceptionUnavailable):
        DFineDetector(strict=True).detect(np.zeros((8, 8, 3), np.uint8))


def test_the_probe_names_every_dead_server(monkeypatch):
    def fake(url, timeout=None):
        if "13185" in url:
            raise OSError("refused")
        return _Resp(b'{"status":"ok"}')

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(PerceptionUnavailable) as e:
        probe_perception_servers({"ram": "http://localhost:13185/ram", "dfine": "http://localhost:13186/dfine"})
    assert "ram" in str(e.value) and "dfine" not in str(e.value)


def test_tag_scene_writes_both_maps_keyed_by_the_floor_step():
    from ascentnav.perception import tag_scene

    class _Obj:
        def __init__(self):
            self.each_step_objects, self.each_step_rooms = {}, {}
            self.this_floor_objects, self.this_floor_rooms = set(), set()

    class _Ram:
        def tags(self, rgb, b64=None):
            return ["shower", "towel"]

    class _Room:
        def classify(self, rgb):
            return "bathroom"

    o = _Obj()
    tag_scene(np.zeros((8, 8, 3), np.uint8), 7, o, _Ram(), _Room())
    assert o.each_step_objects[7] == ["shower", "towel"] and o.each_step_rooms[7] == "bathroom"
    assert o.this_floor_objects == {"shower", "towel"} and o.this_floor_rooms == {"bathroom"}


def test_map_to_room_is_extract_room_categories():
    """`ascent/utils.py:209-229`: the first of the top-5 with a mapping into a
    reference room, else the top-1 class."""
    from osg.perception.room_classifier import map_to_room

    direct = {"shower": "bathroom", "bedchamber": "bedroom"}
    rooms = ["bathroom", "bedroom"]
    assert map_to_room(["attic", "bedchamber", "shower"], direct, rooms) == "bedroom"
    assert map_to_room(["attic", "cellar"], direct, rooms) == "attic"


def test_filter_depth_fills_only_the_zeros():
    """F10: on habitat's normalised depth with `recover_nonzero`, every
    non-zero pixel is restored verbatim and every zero is filled from its
    neighbours."""
    from ascentnav.depth_filter import filter_depth

    d = np.full((40, 40), 0.6, np.float32)
    d[10:14, 10:14] = 0.0                            # a hole
    d[20, 20] = 0.03                                 # a tiny non-zero stays
    out = filter_depth(d.copy(), blur_type=None)
    assert out[10:14, 10:14].min() > 0.0, "the hole was filled"
    assert out[20, 20] == pytest.approx(0.03)
    assert np.allclose(out[d != 0], d[d != 0])


def test_the_value_map_is_built_with_the_references_fusion_mode():
    from .test_ascentnav_stairs import _agent

    a = _agent()
    assert a.value_map._use_max_confidence is False


def test_the_maps_see_filtered_depth_and_the_mover_sees_metres(monkeypatch):
    from .test_ascentnav_stairs import _agent, _wall_frame

    a = _agent()
    a._done_initializing = True
    seen = {}

    def spy_update(depth, *args, **kw):
        seen["min"] = float(depth.min())

    a.obstacle_map.update_map = spy_update
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m=3.0)
    f.depth[0:5, 0:5] = 0.0                           # no-hit pixels
    a.act(f)
    assert seen["min"] > 0.0, "zeros were filled before the obstacle map saw them"
