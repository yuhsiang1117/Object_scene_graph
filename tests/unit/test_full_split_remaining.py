"""The resume set for a long run.

`scripts/run_full_split.sh` restarts the 2000-episode run whenever it dies, and
the only thing standing between that and a corrupted result is this: the
remaining set must never re-run a scored episode or skip an unscored one, from
whatever state a killed process left behind.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[2] / "scripts" / "full_split_remaining.py"
spec = importlib.util.spec_from_file_location("full_split_remaining", MOD)
fsr = importlib.util.module_from_spec(spec)
sys.modules["full_split_remaining"] = fsr
spec.loader.exec_module(fsr)


def _content(tmp_path, scenes) -> str:
    """A content dir shaped like habitat's: one gz per scene."""
    d = tmp_path / "content"
    d.mkdir()
    for scene, count in scenes.items():
        payload = {"episodes": [{"scene_id": f"hm3d/x/{scene}"} for _ in range(count)]}
        with gzip.open(d / f"{scene.split('.')[0]}.json.gz", "wt") as f:
            json.dump(payload, f)
    return str(d)


def _run_dir(tmp_path, name, records) -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(d)


def _rec(scene, i):
    return {"scene": scene, "episode_id": str(i), "success": 1.0, "spl": 0.5}


def test_the_remaining_set_is_everything_not_yet_scored(tmp_path):
    c = _content(tmp_path, {"a.glb": 3, "b.glb": 2})
    d = _run_dir(tmp_path, "run", [_rec("a.glb", 0), _rec("b.glb", 1)])
    out = fsr.every_episode(c)
    done = fsr.scored([d])
    rem = [f"{s}:{i}" for s, i in out if (s, i) not in done]
    assert rem == ["a.glb:1", "a.glb:2", "b.glb:0"]


def test_records_from_several_runs_all_count_as_done(tmp_path):
    """The run was resumed into a second directory; both must be honoured or
    the resume re-runs the first leg."""
    c = _content(tmp_path, {"a.glb": 4})
    d1 = _run_dir(tmp_path, "leg1", [_rec("a.glb", 0)])
    d2 = _run_dir(tmp_path, "leg2", [_rec("a.glb", 2)])
    done = fsr.scored([d1, d2])
    rem = [f"{s}:{i}" for s, i in fsr.every_episode(c) if (s, i) not in done]
    assert rem == ["a.glb:1", "a.glb:3"]


def test_a_truncated_final_line_costs_one_episode_not_the_run(tmp_path):
    """A process killed mid-write leaves a half-written record. That episode is
    simply unscored and gets run again; the file must still parse."""
    c = _content(tmp_path, {"a.glb": 3})
    d = tmp_path / "run"
    d.mkdir()
    good = json.dumps(_rec("a.glb", 0))
    (d / "episodes.jsonl").write_text(good + "\n" + good[:20])
    done = fsr.scored([str(d)])
    assert done == {("a.glb", "0")}


def test_a_missing_directory_is_not_an_error(tmp_path):
    """The resume dir does not exist before the first attempt."""
    assert fsr.scored([str(tmp_path / "nope")]) == set()


def test_an_empty_content_dir_fails_loudly(tmp_path):
    """Silently deriving 'nothing remaining' from a bad path would end the
    supervisor as if the run had finished."""
    empty = tmp_path / "content"
    empty.mkdir()
    with pytest.raises(SystemExit):
        fsr.every_episode(str(empty))


def test_episode_ids_are_compared_as_strings(tmp_path):
    """habitat's episode_id is a string; a record that stored it as an int must
    still match, or every episode looks unscored."""
    c = _content(tmp_path, {"a.glb": 2})
    d = _run_dir(tmp_path, "run", [{"scene": "a.glb", "episode_id": 0, "success": 1.0}])
    done = fsr.scored([d])
    rem = [f"{s}:{i}" for s, i in fsr.every_episode(c) if (s, i) not in done]
    assert rem == ["a.glb:1"]
