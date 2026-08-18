from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from osg.perception.detector import YoloeDetector


def test_yoloe_resolves_text_encoder_beside_weights(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weights = weights_dir / "yoloe.pt"
    weights.write_bytes(b"checkpoint")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    observed = []

    class _Inner:
        def float(self):
            return self

        def to(self, device):
            return self

    class _YOLOE:
        def __init__(self, checkpoint):
            self.model = _Inner()
            self.predictor = object()

        def get_text_pe(self, classes):
            observed.append(Path.cwd())
            return "embeddings"

        def set_classes(self, classes, embeddings):
            pass

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=_YOLOE))
    detector = YoloeDetector(str(weights), device="cpu")
    detector.set_vocabulary(["bowl"])

    assert observed == [weights_dir]
    assert Path.cwd() == run_dir
