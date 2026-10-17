"""The VLM's per-call log (`VLMVerifier.call_log`, episodes.jsonl
`vlm_call_log`): every call's role, target and verdict, with what the caller
said it was about, so the VLM's decisions can be scored against ground truth."""
from types import SimpleNamespace

import numpy as np

from osg.verification.verifier import VLMVerifier, note_vlm


class _Client:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail = reply, fail

    def chat(self, system, user, images=None, json_response=False):
        if self.fail:
            raise RuntimeError("down")
        return self.reply


def _track():
    return SimpleNamespace(id=7, ellipsoid=SimpleNamespace(center=np.array([1.0, 0.5, -2.0])))


def test_a_candidate_call_logs_its_context_and_verdict():
    v = VLMVerifier(_Client({"category": "bowl", "confidence": 0.9}))
    note_vlm(v, "candidate", 42, _track())
    assert v.verify_bbox(np.zeros((8, 8, 3), np.uint8), np.array([1, 1, 5, 5]), "bowl")
    (entry,) = v.call_log
    assert entry == {"kind": "identity", "target": "bowl", "verdict": True, "role": "candidate",
                     "step": 42, "track_id": 7, "xyz": [1.0, 0.5, -2.0]}


def test_the_context_is_consumed_by_one_call():
    v = VLMVerifier(_Client({"category": "cup", "confidence": 0.9}))
    note_vlm(v, "candidate", 1, _track())
    v.verify_bbox(np.zeros((8, 8, 3), np.uint8), np.array([1, 1, 5, 5]), "bowl")
    v.verify_bbox(np.zeros((8, 8, 3), np.uint8), np.array([1, 1, 5, 5]), "bowl")
    assert v.call_log[0]["verdict"] is False and v.call_log[0]["step"] == 1
    assert "step" not in v.call_log[1]


def test_a_failed_call_is_logged_without_a_verdict():
    v = VLMVerifier(_Client(fail=True))
    note_vlm(v, "absence_path_consumed", 9, _track())
    assert v.verify_still_there(np.zeros((32, 32, 3), np.uint8), np.array([4, 4, 20, 20]),
                                "bowl") is None
    assert v.call_log[0]["kind"] == "still_there" and v.call_log[0]["verdict"] is None


def test_note_on_a_stub_verifier_is_a_no_op():
    note_vlm(SimpleNamespace(), "candidate", 3, _track())
