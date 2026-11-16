"""Replay a recorded run to the start of each climb attempt, then let several
on-stairs controllers climb from that same state (S87).

A 21-episode run cannot separate two climb controllers (S85: three rounds of
M2c, every one inside the noise, because which storey an episode ends on is
decided by small differences early in the climb). This removes everything
but the climb. For every episode of `<run>` with a climb attempt:

  1. the agent is rebuilt with the run's config and FED THE RECORDED ACTIONS
     (`step_trace[k].act`) while it acts on the same frames, so its maps, its
     stair state and the simulator's pose are the run's at every step;
  2. at the step an attempt starts, the agent and the simulator's agent state
     are snapshotted, and each arm -- a set of `agent.search.*` switches put
     on a copy -- acts freely until that attempt is over (the stair state
     machine closes it), it STOPs, or `--cap` steps pass;
  3. the snapshot is restored and the replay goes on to the next attempt.

Per attempt and arm: whether the attempt ended 0.5 m (M2b's line) away from
where it began IN ITS OWN DIRECTION ("went its way"; ending as far the other
way -- a "down" attempt the reference walked back up to the storey it came
from -- is counted apart), the height reached, the steps it took, and how it
ended. Arms
are paired on the same attempts; `ref` (no switches) is the control and
should reproduce the recorded outcome. The models and the LLM are queried as
in the run, so the perception servers must be up.

    python scripts/probe_stair_climb.py outputs/s83_baseline_v0 --out outputs/s87_probe \\
        --override +experiment=ascentnav
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {
    "ref": {},
    "mount": {"climb_mount_m": 1.0},
    "heading": {"climb_heading_memory": True},
    "mount+heading": {"climb_mount_m": 1.0, "climb_heading_memory": True},
    "mount+heading+free": {"climb_mount_m": 1.0, "climb_heading_memory": True,
                           "climb_carrot_free": True},
}
LEFT_LEVEL_M = 0.5


def attempt_starts(trace: list) -> list:
    """Steps at whose action a climb attempt began (`climb` goes 0 -> 1/2)."""
    out = []
    for k, t in enumerate(trace):
        prev = trace[k - 1].get("climb", 0) if k else 0
        if not prev and t.get("climb"):
            out.append(k)
    return out


def recorded_outcome(trace: list, k: int) -> dict:
    """The run's own attempt from step k: the height when `climb` returns to 0."""
    y0 = trace[k]["h"]
    j = k
    while j < len(trace) and trace[j].get("climb"):
        j += 1
    y1 = trace[min(j, len(trace) - 1)]["h"]
    d = int(trace[k].get("climb") or 0)
    return {"steps": j - k, "dy": round(y1 - y0, 3), "dir": d,
            "left_level": abs(y1 - y0) >= LEFT_LEVEL_M, "went": _sign(d) * (y1 - y0) >= LEFT_LEVEL_M}


def _sign(direction: int) -> float:
    """+1 for an attempt up (`climb` 1), -1 down (2)."""
    return 1.0 if direction == 1 else -1.0 if direction == 2 else 0.0


# ------------------------------------------------------------------ snapshot

def _shared_objects(root, extra=()) -> dict:
    """Objects a copy must SHARE rather than duplicate: networks, clients,
    threads, the simulator. Everything else (maps, stair state, the point-goal
    policy's hidden state) is copied."""
    import concurrent.futures
    import queue
    import threading
    import types

    import torch

    lock_types = (type(threading.Lock()), type(threading.RLock()), threading.Condition,
                  threading.Event, threading.Thread, concurrent.futures.Executor,
                  concurrent.futures.Future, queue.Queue, types.ModuleType)
    memo, seen, stack = {}, set(), [root]
    for obj in extra:
        memo[id(obj)] = obj
    while stack:
        obj = stack.pop()
        if id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, bool, np.ndarray)):
            continue
        seen.add(id(obj))
        mod = type(obj).__module__ or ""
        if (isinstance(obj, (torch.nn.Module, torch.jit.ScriptModule) + lock_types)
                or mod.split(".")[0] in ("habitat", "habitat_sim", "magnum", "requests", "httpx",
                                         "urllib3", "socket", "ollama", "openai")
                or callable(obj) and not hasattr(obj, "__dict__")):
            memo[id(obj)] = obj
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
            if hasattr(obj, "__self__"):                   # a bound method: share its owner if it is shared
                stack.append(obj.__self__)
    return memo


def snapshot(agent, env, shared: tuple) -> tuple:
    sim = env.env.sim
    state = copy.deepcopy(sim.get_agent(0).get_state())
    counters = (env.env._elapsed_steps, env.env._episode_over, env._frame_id)
    return copy.deepcopy(agent, _shared_objects(agent, shared)), state, counters


def restore_sim(env, state, counters) -> None:
    env.env.sim.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
    env.env._elapsed_steps, env.env._episode_over, env._frame_id = counters


def apply_arm(agent, switches: dict) -> None:
    """Put an arm's `agent.search.*` switches on an agent copy."""
    from navigation.search.climb import HeadingMemory

    st = agent.stairs
    st.mount_m = float(switches.get("climb_mount_m", 0.0))
    st.heading_memory = HeadingMemory() if switches.get("climb_heading_memory") else None
    agent._climb_carrot_free = bool(switches.get("climb_carrot_free", False))


def free_run(agent, env, frame, cap: int, steps_log: bool = False, direction: int = 0) -> dict:
    """Let `agent` act from `frame` until the climb attempt it starts is over."""
    y0 = float(frame.camera_position[1])
    log = []
    before = dict(agent.stats)
    hs, started, reason, first = [y0], False, "cap", None
    steps = 0
    for steps in range(1, cap + 1):
        action = agent.act(frame)
        if steps_log:
            st = agent.stairs
            hm = st.heading_memory
            p = frame.camera_position
            log.append([action[:6], round(float(p[1]) - y0, 2), round(float(p[0]), 2), round(float(p[2]), 2),
                        int(st.reach_stair), int(st.reach_stair_centroid), int(st.climbing),
                        None if hm is None else hm.goal])
        if first is None:
            first = (action, agent._state)
        if agent.stairs.climbing:
            started = True
        if action == "stop":
            reason = "stop"
            break
        frame = env.step(action)
        hs.append(float(frame.camera_position[1]))
        if started and not agent.stairs.climbing:
            reason = "over"
            break
        if not started and steps >= 3:
            reason = "never_started"
            break
    # the attempt's end is judged on the frame the agent acted on last
    y1 = hs[-1]
    delta = {k: v - before.get(k, 0) for k, v in agent.stats.items()
             if isinstance(v, (int, float)) and v != before.get(k, 0)
             and (k.startswith("climb") or k.startswith("stair") or k in ("floor_switches",))}
    return {"started": started, "first": first, "steps": steps, "reason": reason,
            "dy": round(y1 - y0, 3), "max_dy": round(float(max(np.abs(np.array(hs) - y0))), 3),
            "left_level": started and reason == "over" and abs(y1 - y0) >= LEFT_LEVEL_M,
            # the attempt's own direction: mL8ThkuaVTM ep 2 has "down" attempts
            # the reference ended 0.74 m UP, back on the storey it came from
            "went": started and reason == "over" and _sign(direction) * (y1 - y0) >= LEFT_LEVEL_M,
            "wrong_way": started and reason == "over" and _sign(direction) * (y1 - y0) <= -LEFT_LEVEL_M,
            "stats": delta, **({"log": log} if steps_log else {})}


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a finished run with eval.behaviour_log=true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--cap", type=int, default=200, help="steps an arm may take on one attempt")
    ap.add_argument("--episodes", nargs="*", default=None, help="scene:episode_id, default every one with a climb")
    ap.add_argument("--steps", action="store_true", help="keep a per-step log in each arm's row")
    ap.add_argument("--override", dest="overrides", action="append", default=[],
                    help="a Hydra override, as for run_eval.py, giving the RUN's config")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs
    from osg.core.profiler import Profiler
    from osg.eval.runner import _episode_uid
    from osg.pipeline.components import (build_agent, build_env, build_run_components,
                                         probe_served_models)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_file = out / "probe.jsonl"
    done = set()
    if rows_file.exists():                                   # resumable, one line per attempt
        for line in open(rows_file):
            r = json.loads(line)
            done.add((r["uid"], r["k"]))

    recorded = {}
    for line in open(Path(args.run) / "episodes.jsonl"):
        r = json.loads(line)
        tr = r.get("step_trace") or []
        starts = attempt_starts(tr)
        if not starts:
            continue
        uid = f"{r['scene']}:{r['episode_id']}"
        key = f"{r['scene'].split('.')[0]}:{r['episode_id']}"
        if args.episodes and key not in args.episodes:
            continue
        recorded[uid] = (tr, starts)
    print(f"{len(recorded)} episodes, {sum(len(s) for _, s in recorded.values())} attempts; arms {args.arms}")

    register_configs()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(args.overrides) + ["eval.behaviour_log=true"])
    probe_served_models(cfg)
    env = build_env(cfg)
    components = build_run_components(cfg, env=env)
    env.env._max_episode_steps = 10 ** 6                    # the arms run past the recorded episode's end
    episodes = {_episode_uid(e): e for e in env.env.episodes}
    shared = tuple(v for k, v in components.items() if k != "pointnav" and v is not None) + (env,)

    for uid, (tr, starts) in recorded.items():
        if all((uid, k) in done for k in starts):
            continue
        t0 = time.time()
        env.env.current_episode = episodes[uid]
        frame = env.reset()
        components["scorer"].reset()
        agent = build_agent(cfg, components, env.target_category(), profiler=Profiler(),
                            nearest_navigable_fn=getattr(env, "nearest_navigable_xy", None))
        agree = n = 0
        for k, t in enumerate(tr):
            if k in starts and (uid, k) not in done:
                base, state, counters = snapshot(agent, env, shared)
                row = {"uid": uid, "k": k, "recorded": recorded_outcome(tr, k),
                       "agree_so_far": round(agree / max(1, n), 3), "arms": {}}
                for name in args.arms:
                    ag = copy.deepcopy(base, _shared_objects(base, shared))
                    apply_arm(ag, ARMS[name])
                    restore_sim(env, state, counters)
                    row["arms"][name] = free_run(ag, env, frame, args.cap, args.steps,
                                                  row["recorded"]["dir"])
                restore_sim(env, state, counters)
                with open(rows_file, "a") as f:
                    f.write(json.dumps(row, default=float) + "\n")
                print(f"{uid} @{k}: recorded dy {row['recorded']['dy']:+.2f} | "
                      + " | ".join(f"{a} {v['reason']} {v['dy']:+.2f} ({v['steps']})" for a, v in row["arms"].items()),
                      flush=True)
            if k >= starts[-1]:
                break
            action = agent.act(frame)
            n += 1
            agree += action == t["act"]
            frame = env.step(t["act"])
        print(f"{uid}: replay agreed on {agree}/{n} actions, {time.time() - t0:.0f} s", flush=True)
    env.close()
    summarise(rows_file, args.arms)


def summarise(rows_file: Path, arms: list) -> None:
    """Per arm: attempts that ended >= 0.5 m away in their own direction
    ("went"), and the other way; paired against `ref` on "went"."""
    from scipy.stats import binomtest

    rows = [json.loads(line) for line in open(rows_file)]
    rows = [r for r in rows if all(a in r["arms"] for a in arms)]
    if rows and "went" not in rows[0]["recorded"]:
        raise SystemExit("rows from before the direction was recorded: re-run the probe")
    print(f"\n{len(rows)} attempts")
    print(f"  recorded run: went its way in {sum(r['recorded']['went'] for r in rows)}")
    for a in arms:
        v = [r["arms"][a] for r in rows]
        print(f"  {a:20s} started {sum(x['started'] for x in v):3d}  went its way {sum(x['went'] for x in v):3d}  "
              f"the other way {sum(x['wrong_way'] for x in v):3d}  steps {sum(x['steps'] for x in v):5d}  "
              f"median {np.median([x['steps'] for x in v]):.0f}  ended by cap {sum(x['reason'] == 'cap' for x in v)}")
    if "ref" in arms:
        ref = [r["arms"]["ref"]["went"] for r in rows]
        print(f"  ref reproduces the recorded outcome in {sum(x == r['recorded']['went'] for x, r in zip(ref, rows))}/{len(rows)}")
        for a in arms:
            if a == "ref":
                continue
            new = [r["arms"][a]["went"] for r in rows]
            b = sum(x and not y for x, y in zip(new, ref))
            c = sum(y and not x for x, y in zip(new, ref))
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            print(f"  {a:20s} vs ref: better {b}, worse {c}, McNemar p = {p:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--summarise":
        summarise(Path(sys.argv[2]), sys.argv[3:] or list(ARMS))
    else:
        main()
