"""scripts/analyze_objectnav_search.py: the per-episode facts its tables are
built from, on traces small enough to check by hand."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_objectnav_search",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_objectnav_search.py")
aos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aos)

CAM = aos.CAMERA_HEIGHT_M


def step(n, xy=(0.0, 0.0), yaw=0.0, base=0.0, state="explore", **kw):
    row = {"n": n, "xy": list(xy), "xy_ep": list(xy), "yaw": yaw, "h": base + CAM, "state": state,
           "act": "move_forward", "moved": 0.25, "floor": 0, "floor_uid": 0, "n_floors": 1,
           "floor_n": 50, "nfront_live": 2, "explored_m2": 10.0, "climb": 0}
    row.update(kw)
    return row


def episode(trace, **kw):
    row = {"scene": "s", "episode_id": "0", "target": "chair", "success": False,
           "steps": len(trace), "distance_to_goal": 5.0, "floor_class": "same_floor",
           "start_y": 0.0, "step_trace": trace}
    row.update(kw)
    return row


OBJ = np.array([[2.0, 0.0, 0.4]])          # (x, -z, height): 2 m ahead of the origin
VPS = np.array([0.0])


def test_a_storey_change_in_belief_only_is_told_from_one_that_moved_the_agent():
    trace = ([step(i) for i in range(10)]
             + [step(10 + i, base=0.4, state="climb", climb=1) for i in range(5)]
             + [step(15 + i, base=0.1, floor_uid=1) for i in range(10)]             # back at the bottom
             + [step(25 + i, base=1.5, state="climb", climb=1, floor_uid=1) for i in range(5)]
             + [step(30 + i, base=2.9, floor_uid=2) for i in range(10)])            # a storey up
    base = np.array([t["h"] for t in trace]) - CAM
    sw = aos.believed_switches(trace, base)
    assert [(s["step"], s["real"]) for s in sw] == [(15, False), (30, True)]
    assert sw[0]["dy"] == pytest.approx(0.1) and sw[1]["dy"] == pytest.approx(2.8)
    # Without the storey fields there is nothing to count.
    assert aos.believed_switches([{"h": CAM, "state": "explore"}], np.zeros(1)) == []


def test_only_a_commit_is_a_stop_on_the_target():
    walk = [step(i, xy=(-5.0, 0.0), yaw=np.pi) for i in range(6)]      # facing away, 7 m off
    commit = episode(walk + [step(6, xy=(-5.0, 0.0), yaw=np.pi, try_nav=1, state="done")])
    assert aos.features(commit, OBJ, VPS)["stop_from"] == "commit"
    stairs = episode(walk + [step(6, xy=(-5.0, 0.0), yaw=np.pi, state="climb")])
    f = aos.features(stairs, OBJ, VPS)
    assert f["stop_from"] == "climb" and f["class"] == "never saw -> STOP"
    out = episode(walk, approach_stop_reason="explored_out")
    assert aos.features(out, OBJ, VPS)["stop_from"] == "explored_out"
    timeout = episode(walk, steps=aos.TIMEOUT_STEP + 1)
    assert aos.features(timeout, OBJ, VPS)["stop_from"] is None


def test_episodic_points_come_back_in_the_world_plane():
    # Episodic frame: origin at (3, 1), heading 90 deg.
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    eps = [np.array([0.25 * i, 0.1 * (i % 3)]) for i in range(12)]
    trace = [step(i, xy=tuple(rot @ e + [3.0, 1.0])) for i, e in enumerate(eps)]
    for t, e in zip(trace, eps):
        t["xy_ep"] = list(e)
    to_world = aos.episodic_to_world(trace)
    assert to_world([1.0, 0.0]) == pytest.approx([3.0, 2.0], abs=1e-6)
    # An agent that never moved gives nothing to fit.
    assert aos.episodic_to_world([step(i) for i in range(5)]) is None


def test_facts_place_weak_boxes_commits_and_give_ups_on_or_off_the_instance():
    trace = [
        step(0, xy=(0.0, 0.0), yaw=0.0, det=0.0, weak=[[0.6, 2.1, 0.1], [0.5, -4.0, 0.0], [0.45]]),
        step(1, xy=(0.2, 0.0), yaw=0.0, det=0.0),
        step(2, xy=(0.4, 0.0), yaw=0.0, try_nav=1, goal_xy=[-4.0, 0.0], state="approach"),
        step(3, xy=(0.6, 0.0), yaw=0.0),                                    # gave that one up
        step(4, xy=(0.8, 0.0), yaw=0.0, try_nav=1, goal_xy=[2.0, 0.3], state="approach", det=0.9),
        step(5, xy=(1.0, 0.0), yaw=0.0),                                    # and the true one
    ]
    row = episode(trace, giveup_log=[[3, "unverified", [0, 0]], [5, "abandon", [0, 0]]])
    d = aos.facts(row, OBJ, VPS)
    assert (d["weak_true"], d["weak_else"], d["weak_best"]) == (1, 1, 0.6)
    assert (d["commit_steps"], d["commit_true"]) == (2, 1)
    assert (d["gave_true"], d["gave_else"]) == (1, 1)
    assert d["in_view_steps"] == 6 and d["fired_in_view"] == 1
    # The same boxes seen from the storey above are not on the instance.
    for t in trace:
        t["h"] += 3.0
    up = aos.facts(row, OBJ, VPS)
    assert up["weak_true"] == 0 and up["commit_true"] == 0 and up["in_view_steps"] == 0


def test_stuck_is_the_share_of_forward_steps_that_went_nowhere():
    trace = [step(i, xy=(0.25 * i, 5.0), moved=0.25 if i % 2 else 0.0) for i in range(10)]
    trace.append(step(10, xy=(2.5, 5.0), act="turn_left", moved=0.0))
    assert aos.facts(episode(trace), OBJ, VPS)["stuck"] == pytest.approx(0.5)


def _target_log(tmp_path, kfs):
    import json
    path = tmp_path / "s_ep0.targets.jsonl"
    path.write_text("\n".join(json.dumps({"kf": i, "frame_id": fid, "tracks": tracks})
                              for i, (fid, tracks) in enumerate(kfs)) + "\n")
    return tmp_path


def _track(tid, xyz, evidence, score=0.5, n_obs=1):
    return {"id": tid, "center": list(xyz), "evidence": evidence, "best_score": score,
            "n_obs": n_obs, "p": 0.82}


def test_a_candidate_is_true_by_where_it_is_and_reaches_a_threshold_at_a_step(tmp_path):
    # OBJ is at plane (2, 0), height 0.4: habitat (x, y, z) = (2, 0.4, 0).
    replay = _target_log(tmp_path, [
        (10, [_track(1, (2.3, 0.5, 0.2), 0.4)]),
        (20, [_track(1, (2.3, 0.5, 0.2), 1.2, score=0.7, n_obs=3), _track(2, (-4.0, 0.5, 0.0), 0.6)]),
        (30, [_track(1, (2.3, 0.5, 0.2), 0.9), _track(2, (-4.0, 0.5, 0.0), 0.6),
              _track(3, (2.0, 3.4, 0.0), 2.0)]),                       # above it, a storey up
    ])
    tracks = {t["id"]: t for t in aos.load_candidates(replay, ("s", "0"), OBJ)}
    assert [tracks[i]["true"] for i in (1, 2, 3)] == [True, False, False]
    assert tracks[1]["first"] == 10 and tracks[1]["evidence"] == 1.2 and tracks[1]["n_obs"] == 3
    assert aos.reached(tracks[1], 1.0) == 20 and aos.reached(tracks[1], 0.3) == 10
    assert aos.reached(tracks[2], 1.0) == -1
    assert aos.load_candidates(replay, ("s", "9"), OBJ) == []


def test_auc_and_the_places_false_tracks_stand_for():
    assert aos.auc([2.0, 3.0], [1.0, 1.5]) == 1.0
    assert aos.auc([1.0], [1.0]) == 0.5
    assert aos.auc([], [1.0]) is None
    near = [{"xy": np.array([0.0, 0.0])}, {"xy": np.array([0.5, 0.0])}, {"xy": np.array([3.0, 0.0])}]
    assert aos.sites(near) == 2 and aos.sites([]) == 0
