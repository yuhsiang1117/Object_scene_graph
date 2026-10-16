"""The profiler's tail percentiles and `scripts/analyze_latency.py`.

A real-time claim is about the slow frames, so the report carries p90/p99 as
values actually measured (nearest rank), and the analyzer reads them -- or
prints "-" for a run written before they existed.
"""
import importlib.util
import json
from pathlib import Path

from osg.core.profiler import Profiler

ROOT = Path(__file__).resolve().parents[2]


def _analyzer():
    spec = importlib.util.spec_from_file_location("analyze_latency", ROOT / "scripts/analyze_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_percentiles_are_measured_values_at_nearest_rank():
    p = Profiler()
    for ms in range(1, 101):                 # 1 .. 100 ms
        p.add("x", ms / 1000.0)
    r = p.report()["x"]
    assert r["p90_ms"] == 91.0 and r["p99_ms"] == 100.0 and r["median_ms"] == 51.0
    one = Profiler()
    one.add("y", 0.5)
    assert one.report()["y"]["p90_ms"] == one.report()["y"]["p99_ms"] == 500.0


def test_the_csv_keeps_its_columns_and_adds_the_tail(tmp_path):
    p = Profiler()
    p.add("control_loop", 0.2)
    p.write_csv(str(tmp_path / "timing.csv"))
    header = (tmp_path / "timing.csv").read_text().splitlines()[0].split(",")
    assert header[:6] == ["module", "count", "mean_ms", "median_ms", "max_ms", "total_s"]
    assert header[6:] == ["p90_ms", "p99_ms"]


def test_the_analyzer_reads_a_run_and_separates_synchronous_model_calls(tmp_path, capsys):
    p = Profiler()
    for _ in range(10):
        p.add("control_loop", 0.5)
        p.add("detector", 0.1)
    p.add("absence_vlm", 20.0)
    (tmp_path / "summary.json").write_text(json.dumps({"timing": p.report()}))
    result = _analyzer().table(_analyzer().load_timing(tmp_path))
    out = capsys.readouterr().out
    assert result["steps"] == 10
    assert "| perception | detector | 1.00 | 100.0 |" in out
    assert "absence_vlm: 1 calls" in out and "2.00 Hz" in out


def test_an_old_run_without_percentiles_prints_a_dash(tmp_path, capsys):
    (tmp_path / "timing.csv").write_text(
        "module,count,mean_ms,median_ms,max_ms,total_s\n"
        "control_loop,4,250.00,240.00,300.00,1.00\n"
        "detector,4,50.00,50.00,60.00,0.20\n")
    _analyzer().table(_analyzer().load_timing(tmp_path))
    assert "| 50.0 | 50.0 | - | 60.0 |" in capsys.readouterr().out
