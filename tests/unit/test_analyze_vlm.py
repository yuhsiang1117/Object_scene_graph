"""`scripts/analyze_vlm.py`: a VLM verdict is scored by the track it was about."""
import importlib.util
from collections import OrderedDict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze_vlm", Path(__file__).resolve().parents[2] / "scripts" / "analyze_vlm.py")
analyze_vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_vlm)


def _record(calls):
    return {"authored_layout": {"target_position": [0.0, 1.0, 0.0],
                                "prior_position": [5.0, 1.0, 0.0]},
            "verify_calls": len(calls), "vlm_call_log": calls}


def test_verdicts_are_scored_against_the_true_and_prior_poses(capsys):
    calls = [
        {"role": "candidate", "verdict": True, "xyz": [0.3, 1.0, 0.2]},       # TRUE, accepted
        {"role": "candidate", "verdict": True, "xyz": [9.0, 1.0, 0.0]},       # FP, accepted
        {"role": "absence_path_consumed", "verdict": False, "xyz": [5.2, 1.0, 0.1]},  # STALE, bare
        {"role": "absence_deadline", "verdict": None, "xyz": [0.0, 1.0, 0.0]},  # failed call
    ]
    out = analyze_vlm.report(OrderedDict(run=[_record(calls)]))["run"]
    assert out["tables"]["identity"] == {"TRUE/yes": 1, "FP/yes": 1}
    assert out["tables"]["absence"] == {"STALE/no": 1, "TRUE/none": 1}
    assert "right 1/2" in capsys.readouterr().out


def test_another_storey_is_not_the_object():
    rec = _record([])
    assert analyze_vlm.truth(rec, [0.0, 4.0, 0.0]) == "FP"
    assert analyze_vlm.truth(rec, [0.0, 1.2, 0.5]) == "TRUE"
