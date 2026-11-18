# Methodology

How this repo measures ObjectNav, what the default pipeline does on each step,
and every way it differs from native ASCENT. Written from the code and the
recorded runs; where a claim could be checked mechanically it was, and the
check is named.

---

## 1. What is being measured

HM3D ObjectNav v1, sensor-only. The agent gets RGB-D and an odometry-style
pose; it never sees the simulator's navmesh, the semantic mesh, or the goal
position. Success and SPL come from habitat's own measures
(`env.get_metrics()`, `src/osg/sim/habitat_env.py:307`), not from anything this
repo computes.

| | |
|---|---|
| dataset | HM3D ObjectNav v1 val (2000 episodes, 36 scenes) |
| primary split | `scenes20_ep0to4` — 20 scenes × episodes 0–4 = 100 episodes |
| success | `distance_to: VIEW_POINTS`, `success_distance 0.1` m |
| budget | 500 steps; the harness forces a STOP at step 499 |
| actions | `stop, move_forward (0.25 m), turn_left/right (30°), look_up/down (30°)` |
| camera | 640×480, 79° HFOV, 0.88 m, depth clipped 0.5–5.0 m |
| body | 0.88 m tall, 0.18 m radius |

**Episodes are pinned by id, not by iterator order.** `eval.episode_ids`
selects exactly which episodes run (`configs/eval/scenes20_ep0to4.yaml`), so
two runs — including a run of native ASCENT in a different harness — score the
same episodes regardless of how each iterates the dataset. Every paired number
in `docs/history/AB_RESULTS.md` depends on this.

---

## 2. The default pipeline

`configs/config.yaml` composes the **S71 arm**: a transcription of ASCENT's
control flow on ASCENT's served perception models. 63.0% SR / 0.36 SPL on the
100-episode split; native ASCENT scores 65.0% / 0.36 on the same episodes.

Five models are served over HTTP from a separate conda env
(`bash scripts/serve_perception.sh`), because BLIP-2's `lavis` and habitat-sim
cannot share an interpreter. Process-per-model is ASCENT's own architecture,
not a workaround around it.

| role | model | port |
|---|---|---|
| target detection | D-FINE (closed-set COCO), conf ≥ 0.8 | 13186 |
| instance masks | MobileSAM, one per detection box | 13183 |
| value map + commit gate | BLIP-2 ITM cosine | 13182 |
| stair mask | GroundingDINO `"stair ."` ≥ 0.60, ANDed with RedNet | 13184 |
| prompt content | RAM++ tags, per step | 13185 |
| frontier planner | Qwen2.5-7B on local ollama | — |

Each step, in this order (`src/navigation/agent.py::_act_inner`):

1. Normalise and hole-fill depth (`filter_depth`, zeros only); compute the
   camera→episodic transform from the episode-start anchor.
2. **Object map** — skipped entirely on a floor's first step. Detect, keep
   target-class detections at 0.8, ingest *every* one with its own SAM mask,
   tag the frame with RAM++ and Places365, build the stair mask.
3. **Obstacle map** — stair state machine, then the map update, frontiers,
   floor bookkeeping.
4. **Value map** — one BLIP-2 cosine for the whole frame.
5. Distance to the nearest in-range point of the target cloud; the goal.
6. **Dispatch**: stairs → pitch levelling → opening scan (13 turn-lefts) →
   explore if there is no goal → otherwise navigate.

Two rules carry most of the behaviour and are easy to get wrong:

- **The commit gate** latches when navigation has already begun on a *prior*
  step, a target detection exists this frame, and the **previous** step's
  BLIP-2 cosine was ≥ 0.15. It is never cleared, including by a failed
  approach. The ordering is load-bearing: the object map is updated before the
  value map, so the gate reads a cosine one step old.
- **The failure path** clears the cloud, burns its cells, resets the navigate
  counter, and returns the *exploration* action on the same step. The gate
  stays latched.

`src/navigation/README.md` lists the fidelity findings (F1–F14) behind these.

---

## 3. What a run produces

```bash
python scripts/run_eval.py eval.behaviour_log=true output_dir=outputs/<run>
```

- `episodes.jsonl` — one record per episode: habitat's `success`/`spl`/
  `distance_to_goal`, step count, `agent_stats` (every mechanism counter),
  `state_log`, `giveup_log`, `approach_stop_reason`, and with
  `behaviour_log=true` a full `step_trace`: per-step pose, pitch, state,
  action, detections and their scores/pixel areas, BLIP-2 cosine, frontier
  count, stair state, whether the forward actually moved.
- `summary.json` — the composed config plus an explicit algorithm field
  inventory, so two runs can be diffed field by field.
- `timing.csv`.

The step trace is what makes the diagnosis in §4 possible; without it a run
yields one number and no explanation.

---

## 4. How claims are established

Three rules, each adopted after the alternative produced a wrong answer.

### 4.1 Score against the dataset's own geometry, not against the agent's beliefs

`scripts/compare_ascent_osg.py` pairs two runs episode by episode and scores
both step traces against the goal positions in the episode JSONs — geometry
neither agent ever sees. The metrics that matter:

| metric | what it isolates |
|---|---|
| **SAW** — some step had a true instance within 3 m and ±40° | did the search ever put the target in frame |
| never-saw → STOP / → timeout | search truncated before the target was reachable |
| conversion given SAW | everything after the target is visible |
| climb-mode steps on same-floor episodes | budget lost to staircases that cannot help |
| earliest STOP, STOPs before step 33 | premature irreversible stops |
| P(success \| committed) | approach quality once the agent commits |

This decomposition is why the S71 rewrite worked. Seventeen prior arms tuned
thresholds and swapped models and all landed at 45–54%, because SR alone
cannot distinguish "stops on the wrong object" from "never found the right
one". The trace said the gap was almost entirely **before** commit: SAW 81 vs
ASCENT's 90, conversion equal.

### 4.2 Pair the episodes and report the paired test

Runs are compared on the same episode ids, with wins/losses and a McNemar
exact test — not two independent success rates. At n=100 a 3-point difference
is inside the noise; several published-looking effects in `docs/history/AB_RESULTS.md`
are ±2 with p > 0.7.

### 4.3 Count the mechanism, not just the outcome

**A null result is uninterpretable unless the mechanism is known to have
fired.** S72 changed one flag, moved zero episodes, and looked like "the
multi-floor prompt does not help". Adding counters showed the branch had been
asked **4 times in 1733 steps**, because three geometric short-circuits ahead
of it absorb 98.3% of frontier decisions. The finding was not about floors at
all — it was that the entire LLM cascade is starved.

Every mechanism therefore carries a counter in `agent_stats`
(`frontier_decisions`, `fd_single`, `fd_force`, `fd_nearby`,
`llm_path_reached`, `multi_floor_asks`, `gate_latched`, `climb_attempt/ok/fail`,
`passive_stair_entry`, `give_up_*`, …), and an arm is only reported once its
counters show the change happened.

### 4.4 Fail loud

Served models raise `PerceptionUnavailable` rather than returning a neutral
value, and all five are probed before habitat loads. A silently unreachable
BLIP-2 returns a cosine of 0, which under ASCENT's gate is an agent that can
never STOP — a 0% run that looks like a bad algorithm. The one exception is a
*transient* fault: `post_json` retries once, because a single HTTP 500 from a
non-reentrant model server used to kill multi-hour runs (it ended a 2000-episode
run at episode 699). A server that stays down still raises.

**One evaluation at a time.** The servers hold per-image state on the model
object and are not reentrant: the lock that once serialised them was a local
patch to the submodule, which is now pinned to upstream unmodified
(`docs/SETUP.md` §6). Two concurrent evals race, and the loser gets HTTP 500.

---

## 5. Differences from native ASCENT

Reference: `relative_work/ascent` (a refactored release of the authors' code).
Everything below was checked mechanically unless marked otherwise.

### 5.1 Verified identical

| | evidence |
|---|---|
| **PointNav mover weights** | 80/80 tensors present in both and **bitwise equal** (`torch.equal` on every tensor) |
| **Planner constants** | all 25 shared constants identical, value for value (sticky 0.3 m / 20 steps, repeated-selection 20, multi-floor ask 60, floor-exp 100, …); this repo adds only two *names* for literals ASCENT inlines (`INITIALIZE_TURNS`, `PASSIVE_STAIR_DETECTION_THRESHOLD`) |
| **`ValueMap`** | identical apart from import paths |
| **Agent embodiment and sensors** | both take habitat's `objectnav_hm3d.yaml` benchmark defaults: 640×480, 79°, 0.88 m, 0.18 m, 30° turns, 0.25 m steps, 500 steps |
| **Success/SPL** | habitat's own measures on both sides |
| **Model servers** | same five modules on the same ports (ASCENT's `BASE_PORT=13181` → 13182–13186) |
| **Single environment** | ASCENT sets `num_environments: 1`; this repo runs one `habitat.Env` |

### 5.2 Substituted, with the reason

| | ASCENT | here | why |
|---|---|---|---|
| **DBSCAN** | `open3d.cluster_dbscan` | `sklearn.cluster.DBSCAN` | same algorithm, same noise convention, used in exactly one place; open3d is a 400 MB dependency for one call |
| **`ObjectPointCloudMap`** | as above | + a `_cluster_dbscan` helper | the only substantive change; rest is import paths |
| **`ObstacleMap`** | mirrored-depth down-stair trigger | same by default (`downstair_detector: ascent`), plus an opt-in `lip` alternative | the reference's own comment concedes the trick is weak; the alternative is a flagged A/B, off by default |
| **Harness** | `habitat_baselines` trainer (`trainer_name: ascent`) | this repo's `src/osg/eval/runner.py` | needed for the episode-id pinning, the step trace, and per-arm artifacts |

A `stair_mask & stair_map` fusion gained an explicit `.astype(bool)` on both
operands — a dtype guard, equivalent for 0/1 masks.

### 5.3 Genuine behavioural differences

1. **Multi-floor LLM prompt.** In ASCENT it is unreachable: `_explore` never
   passes `floor_num`, so the `>1` branch cannot run and the `-100/-200`
   sentinels never occur. Here it is ported and gated behind
   `exploration.llm_multi_floor` (**default false = the reference's effective
   behaviour**). Measured with it on: null, and the branch fires ~4 times per
   1700 steps (S72).
2. **Stair-disable burn.** `_disable_stair_and_reset_state` zeroes the climb
   flag *before* testing it, so the branch that burns a failed staircase off
   the map never executes and the staircase is retried immediately. Ported
   as-is, dead branch included; `+agent.stair_disable_burns_map=true` runs the
   branch the code was evidently written for (the `+` is needed: the flag is
   read with `getattr` and has no field in `AgentConfig`).
3. **`_pointnav` stop radius.** The reference's radius early-return is inert —
   `stop=True` is never passed at any call site — so the mover is always asked
   with radius 0 and a network STOP comes back for the caller to interpret.
   Reproduced, but untidily: `_pointnav` still *accepts* a `stop_radius`
   argument, callers still pass `self.stop_radius` (0.9) at four sites, and the
   body ignores it and hands the driver `0.0` (`agent.py:970`). The behaviour is
   the reference's; the dead parameter is a wart worth removing.
4. **Map anchor.** ASCENT anchors its maps at the episode start (GPS); an
   earlier version of this repo anchored at the habitat world origin, which
   with a 1600 px map at 20 px/m could overflow. Now anchored at the episode
   start, matching ASCENT. Both frames are logged in the trace.
5. **Instrumentation.** A `BehaviourLog`, a per-step `step_trace`, and
   `agent_stats` counters have no ASCENT counterpart. They are write-only —
   no decision reads them.
6. **An inert `SceneGraph`** is constructed per episode and never read
   (`self.scene_graph` is assigned twice in `agent.py` and referenced nowhere
   else). It is a hook for OSG's other agents, not part of this arm.
7. **Removed OSG-only machinery.** The VLM verifier, commit gate, arrival
   gate, weak memory, soft give-up and displacement escape have no ASCENT
   counterpart and were deleted from this arm. `AgentConfig` still carries
   flags read only by OSG's other policies.

### 5.4 Deviations shared by both sides

These affect the reference number too, so they do **not** appear as a gap in
the comparison — but they separate both from ASCENT's published setup.

- **The LLM is quantised.** ASCENT's launcher serves Qwen2.5-7B in bf16
  (`model_api.qwen25_out`, ~15 GB). Neither side does here: both call
  `qwen2.5:7b` on ollama, **Q4_K_M, 4.7 GB**, through a drop-in that serves
  ASCENT's exact endpoint. The reference run's own log records
  `Routing ASCENT's Qwen2.5 calls to http://ollama:11434 (qwen2.5:7b)`. Given
  S72/S73 — the LLM touches 1.7% of frontier decisions, and opening that to
  9.3% moves no episodes — this is unlikely to matter, but it is not the
  paper's configuration.
- **One 24 GB GPU** hosts habitat, RedNet and five model servers, so the
  servers are shared between the two stacks rather than duplicated.
- **The reference number is 65.0% on 100 episodes**, measured here, not the
  paper's 63% on the full 2000-episode split. Compare like with like: this
  repo has no finished full-split run yet. `+experiment=final_sensor` is that
  run, `scripts/run_full_split.sh` supervises it into `outputs/full_final*`, and
  `docs/history/AB_RESULTS.md` still lists it as pending.

---

## 6. Reproducing

```bash
bash scripts/serve_perception.sh                 # five servers, once
python scripts/run_eval.py eval.behaviour_log=true output_dir=outputs/my_run
python scripts/compare_ascent_osg.py data/reference/ascent_behaviour_100 outputs/my_run
```

**The reference trace is not in this checkout.** `data/reference/ascent_behaviour_100/` was
meant to be committed beside the code (commit `11e0b4a` says so), but `data/*` was ignored
and it never was, and the recorder patch that produced it is no longer applied to the
submodule, so it cannot be regenerated. `.gitignore` now admits `data/reference/`: whoever
holds the copy from the S71 machine should put it back there and commit it.

`pytest tests/unit -q` (~50 s, no GPU) pins the transcription: every test in
`tests/unit/test_ascentnav_*.py` names the reference line it was transcribed
from, and `tests/unit/golden/` pins the composed config of every preset so a
default cannot move unnoticed.

`docs/history/AB_RESULTS.md` is the running log — every arm, its mechanism counters,
its paired result, and the ones that measured null.
