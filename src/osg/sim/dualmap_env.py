"""Our stack, on DualMap's released benchmark, under DualMap's own rules.

`ycb_authored` runs the authored layouts through episodes this repository
generates: it samples its own starts, scores at its own success distance, and
answers questions about our system. It cannot answer "how do the two systems
compare", because nothing in it is shared with the other system's run.

This mode is the other half. Every input that defines a trial is taken from the
released benchmark or from the DualMap run already measured against it:

  the 186 queries and the layout injected for each come from
  `osg.eval.dualmap_release`, the same module the DualMap harness imports;

  the start pose comes from that DualMap run's own per-trial record, so both
  systems answer the same query from the same place in the same world;

  and a stop is scored by `distance_to_target` at `SUCCESS_DISTANCE_M`, within
  three attempts -- DualMap's definition, in DualMap's code, not ours.

What is NOT shared is the agent's embodiment: our navmesh is cut for a 0.88 m
agent and DualMap drove a 1.5 m one, so a start pose from the other system is
snapped onto this navmesh (the offset is recorded per trial) and the optimum
path is recomputed here rather than copied. Both optima are written down so the
difference is measurable instead of assumed.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.config.ycb import YCB_TARGET_LABELS
from ..core.paths import hm3d_scene_root
from ..eval.dualmap_release import (
    HANDLE_TO_QUERY,
    MAX_ATTEMPTS,
    OBJECTS_ROOT,
    RELEASE_ROOT,
    SUCCESS_DISTANCE_M,
    distance_to_target,
    protocol,
    shortest_success_path,
    swapped_trial_id,
    trial_targets,
)
from .habitat_env import HabitatObjectNavEnv, make_objectnav_config
from .ycb_env import inject_layout_objects
from .ycb_layouts import AuthoredLayout, AuthoredObject, YCBLayoutError


# The benchmark asks for an OBJECT. Which string to hand a text-conditioned
# detector to find that object is part of the system, not part of the query, and
# the two systems do not agree on it: DualMap asks MobileCLIP for "pitcher",
# while the probe behind `YCB_TARGET_LABELS` measured "pitcher" at 0.00 on this
# asset at every resolution and "blue plastic pitcher" at 0.71 (see
# osg/perception/vocabulary.py). Three of the eight YCB queries are renamed
# here; the other five and every HM3DSem class are passed through unchanged.
# The trial keeps the released query as its identity -- this only chooses what
# our detector is asked, and the mapping predates this benchmark run.
QUERY_TO_LABEL = {
    query: YCB_TARGET_LABELS[handle] for handle, query in HANDLE_TO_QUERY.items()
}


def agent_query(query: str) -> str:
    """The name OUR detector answers to for the object this query names."""
    return QUERY_TO_LABEL.get(query, query)


def released_layout(scene: str, layout_path: Path) -> AuthoredLayout:
    """Read a released layout file as an injectable layout.

    The released JSON is the collector's own output before this repository
    touched it: no `authoring` block, and scene paths still pointing at the
    authors' machine. `load_authored_layout` validates all of that and is the
    right loader for our authored tree; here the file is the benchmark itself,
    so it is read as-is and the assets are resolved by scene name -- exactly
    what the DualMap harness does with the same file.
    """
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    mapping = {int(key): str(value) for key, value in data["id_handle_mapping"].items()}
    stem = scene.split("-", 1)[1]
    scene_mesh = hm3d_scene_root() / f"val/{scene}/{stem}.basis.glb"
    dataset_config = hm3d_scene_root() / "hm3d_annotated_basis.scene_dataset_config.json"
    objects: List[AuthoredObject] = []
    for entry in data["objects"]:
        semantic_id = int(entry["semantic_id"])
        handle = mapping.get(semantic_id)
        if handle is None:
            raise YCBLayoutError(f"{layout_path}: semantic ID {semantic_id} has no handle")
        objects.append(
            AuthoredObject(
                semantic_id=semantic_id,
                handle=handle,
                label=str(YCB_TARGET_LABELS.get(handle, handle)).strip().lower(),
                translation=tuple(float(x) for x in entry["translation"]),
                rotation=tuple(float(x) for x in entry["rotation"]),
                # The released files carry no anchor metadata; the anchor is
                # what the SPLIT means, not something a trial needs at runtime.
                anchor_object_id="",
                anchor_category="",
            )
        )
    if not objects:
        raise YCBLayoutError(f"{layout_path}: layout contains no placed objects")
    return AuthoredLayout(
        scene_name=scene,
        layout_type="released",
        layout_index=None,
        layout_path=layout_path,
        layout_relative_path=layout_path.name,
        layout_sha256="",
        layout_id=layout_path.stem,
        scene_mesh=scene_mesh,
        scene_mesh_relative_path=f"val/{scene}/{stem}.basis.glb",
        scene_dataset_config=dataset_config,
        scene_dataset_config_relative_path="hm3d_annotated_basis.scene_dataset_config.json",
        objects_dir=OBJECTS_ROOT,
        id_handle_mapping=tuple(sorted(mapping.items())),
        objects=tuple(sorted(objects, key=lambda obj: obj.semantic_id)),
    )


def load_reference(run_root: Path) -> Dict[str, Dict[str, Any]]:
    """The measured DualMap run that supplies each trial's start pose."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_root.glob("trials/*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[str(record["trial_id"])] = record
        # Under an asset swap the trial is renamed after its new query; the
        # start pose is the released trial's, so alias it under the new id.
        out.setdefault(swapped_trial_id(str(record["trial_id"])), record)
    if not out:
        raise FileNotFoundError(f"no DualMap trial results under {run_root}")
    return out


def select_trials(cfg) -> List[Dict[str, Any]]:
    """The protocol, narrowed by the run's selectors and nothing else."""
    dm = cfg.dualmap
    scenes = [str(s) for s in (dm.scenes or ["*"])]
    conditions = [str(c) for c in (dm.conditions or [])]
    queries = [str(q) for q in (dm.queries or [])]
    wanted = {str(t) for t in (dm.trial_ids or [])}
    trials = []
    for trial in protocol():
        if wanted:
            if trial["trial_id"] not in wanted:
                continue
        else:
            if scenes != ["*"] and trial["scene"] not in scenes:
                continue
            if conditions and trial["condition"] not in conditions:
                continue
            if queries and trial["query"] not in queries:
                continue
        trials.append(trial)
    if not trials:
        raise ValueError("dualmap trial selection matched no trial")
    if int(dm.max_trials) > 0:
        trials = trials[: int(dm.max_trials)]
    return trials


def start_rotation(trial_id: str, seed: int) -> List[float]:
    """A deterministic start heading.

    The DualMap run recorded where the agent started but not which way it
    faced, and the released benchmark publishes neither. Position is therefore
    matched exactly and heading is drawn from the trial id, so the choice is
    reproducible, independent of trial order, and not tuned per trial.
    """
    rng = random.Random(f"{seed}:{trial_id}")
    yaw = rng.uniform(-math.pi, math.pi)
    return [0.0, float(math.sin(yaw / 2.0)), 0.0, float(math.cos(yaw / 2.0))]


class DualMapProtocolEnv(HabitatObjectNavEnv):
    """ObjectNav-compatible environment over the released DualMap trials."""

    def __init__(self, cfg) -> None:
        import habitat

        self.cfg = cfg
        if int(cfg.eval.attempts) != MAX_ATTEMPTS:
            raise ValueError(
                f"the released protocol allows {MAX_ATTEMPTS} attempts per query; "
                f"eval.attempts={cfg.eval.attempts}"
            )
        self.trials = select_trials(cfg)
        self.reference = load_reference(Path(str(cfg.dualmap.reference_run)))
        missing = [t["trial_id"] for t in self.trials if t["trial_id"] not in self.reference]
        if missing:
            raise ValueError(
                f"{len(missing)} selected trials have no start pose in "
                f"{cfg.dualmap.reference_run} (first: {missing[0]})"
            )
        self._targets = {
            trial["trial_id"]: trial_targets(trial) for trial in self.trials
        }
        self._layouts: Dict[str, AuthoredLayout] = {}
        for trial in self.trials:
            key = f"{trial['scene']}/{trial['condition']}/{trial['layout']}"
            if key not in self._layouts:
                self._layouts[key] = released_layout(trial["scene"], Path(trial["layout_path"]))

        dataset = self._make_dataset()
        self._hab_cfg = make_objectnav_config(cfg)
        self.env = habitat.Env(config=self._hab_cfg, dataset=dataset)
        from ..core.types import CameraIntrinsics

        self.intrinsics = CameraIntrinsics.from_hfov(
            cfg.eval.hfov_deg, cfg.eval.rgb_width, cfg.eval.rgb_height
        )
        self._frame_id = 0
        self._follower = None
        self._action_name = {value: key for key, value in self.ACTIONS.items()}
        self._navmesh_goal_radius = float(cfg.agent.navmesh_goal_radius)
        self.snap_on_agent_island = bool(getattr(cfg.agent, "navmesh_snap_on_agent_island", False))
        self._active_objects: List[Any] = []
        self._trial: Dict[str, Any] = {}
        self._attempts: List[Dict[str, Any]] = []
        self._start_snap: Dict[str, Any] = {}
        self._shortest = math.inf
        self._travelled = 0.0
        self._previous = np.zeros(3, dtype=float)

    # ------------------------------------------------------------- dataset

    def _make_dataset(self):
        from habitat.core.simulator import AgentState
        from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
        from habitat.tasks.nav.object_nav_task import (
            ObjectGoal,
            ObjectGoalNavEpisode,
            ObjectViewLocation,
        )

        dataset = ObjectNavDatasetV1()
        dataset.episodes = []
        # Habitat indexes episodes by the category string the episode
        # carries, which is the name our detector is asked for.
        labels = sorted({agent_query(trial["query"]) for trial in self.trials})
        dataset.category_to_task_category_id = {
            label: index for index, label in enumerate(labels)
        }
        dataset.category_to_scene_annotation_category_id = dict(
            dataset.category_to_task_category_id
        )
        dataset.goals_by_category = {}

        seed = int(self.cfg.dualmap.start_yaw_seed)
        for trial in self.trials:
            trial_id = trial["trial_id"]
            reference = self.reference[trial_id]
            layout = self._layouts[
                f"{trial['scene']}/{trial['condition']}/{trial['layout']}"
            ]
            targets = self._targets[trial_id]
            start = [float(x) for x in reference["start_position"]]
            goals = []
            for index, (centre, half) in enumerate(targets):
                # Habitat's own success/SPL measures need a view point to exist.
                # They are NOT the protocol's metrics -- `attempt_scored` and the
                # trial record below are -- so this is the object itself lifted
                # to the start height, never a claim about where to stand.
                goals.append(
                    ObjectGoal(
                        position=[float(centre[0]), float(centre[1]), float(centre[2])],
                        object_id=f"{trial_id}#{index}",
                        object_name=str(trial["query"]),
                        object_name_id=index,
                        object_category=agent_query(trial["query"]),
                        view_points=[
                            ObjectViewLocation(
                                agent_state=AgentState(
                                    position=[
                                        float(centre[0]), start[1], float(centre[2])
                                    ],
                                    rotation=[0.0, 0.0, 0.0, 1.0],
                                ),
                                iou=None,
                            )
                        ],
                    )
                )
            info = {
                "ycb": {
                    "scene": trial["scene"],
                    "layout_id": layout.layout_id,
                    "layout_type": trial["condition"],
                    "condition": trial["condition"],
                    "trial_id": trial_id,
                    "query": trial["query"],
                    "agent_query": agent_query(trial["query"]),
                    "layout": trial["layout"],
                    "layout_path": trial["layout_path"],
                    "layout_sha256": trial["layout_sha256"],
                    # Diagnostics (`gt_*`) track ONE object; the protocol scores
                    # against every instance, which `target_positions` keeps.
                    "target_position": [float(x) for x in targets[0][0]],
                    "target_positions": [[float(v) for v in c] for c, _ in targets],
                    "target_half_extents": [[float(v) for v in h] for _, h in targets],
                    "reference_start": start,
                    "reference_shortest_success_path_m": reference.get(
                        "shortest_success_path_m"
                    ),
                }
            }
            dataset.episodes.append(
                ObjectGoalNavEpisode(
                    episode_id=trial_id,
                    scene_id=str(layout.scene_mesh),
                    scene_dataset_config=str(layout.scene_dataset_config),
                    additional_obj_config_paths=[str(layout.objects_dir)],
                    start_position=start,
                    start_rotation=start_rotation(trial_id, seed),
                    goals=goals,
                    object_category=agent_query(trial["query"]),
                    info=info,
                )
            )
            dataset.goals_by_category[dataset.episodes[-1].goals_key] = goals
        return dataset

    # ------------------------------------------------------------- episode

    def reset(self):
        self.nav_reasons.clear()
        self.env.reset()
        info = (getattr(self.current_episode, "info", None) or {}).get("ycb", {})
        trial_id = str(info.get("trial_id"))
        self._trial = dict(info)
        self._attempts = []

        layout = self._layouts[
            f"{info['scene']}/{info['condition']}/{info['layout']}"
        ]
        self._active_objects = inject_layout_objects(self.env.sim, layout)

        # The other system's start pose was drawn on the other system's navmesh.
        # Snapping is what makes it usable here; the offset is recorded so a
        # trial where the two embodiments disagree about the floor is visible
        # rather than silently a different trial.
        sim = self.env.sim
        requested = np.asarray(info["reference_start"], dtype=np.float32)
        snapped = np.asarray(sim.pathfinder.snap_point(requested), dtype=float)
        if not np.isfinite(snapped).all():
            snapped = np.asarray(requested, dtype=float)
        state = sim.get_agent_state()
        sim.set_agent_state(np.asarray(snapped, dtype=np.float32), state.rotation)
        self._start_snap = {
            "requested": [float(x) for x in requested],
            "snapped": [float(x) for x in snapped],
            "offset_m": float(np.linalg.norm(snapped - np.asarray(requested, dtype=float))),
        }

        targets = self._targets[trial_id]
        self._shortest = shortest_success_path(sim.pathfinder, snapped, targets)

        observations = sim.get_observations_at()
        if observations is None:
            raise RuntimeError("failed to refresh observations after layout injection")
        self._frame_id = 0
        self._travelled = 0.0
        self._previous = np.asarray(snapped, dtype=float)
        return self._to_frame(observations)

    def step(self, action: str):
        # A STOP is the act being scored, and the last one is never routed
        # through `attempt_scored` -- habitat ends the episode on it, so the
        # protocol has to catch it here, before the pose is gone.
        if action == "stop":
            self._record_stop()
        frame = super().step(action)
        here = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
        self._travelled += float(np.linalg.norm((here - self._previous)[[0, 2]]))
        self._previous = here
        return frame

    # -------------------------------------------------------------- scoring

    def _record_stop(self) -> bool:
        """Score one stop by the released rule and keep it.

        Deliberately not our own criterion: the released benchmark counts a
        query answered when the agent stops within one metre of any instance of
        the queried object, and that is what this records.
        """
        position = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
        targets = self._targets[str(self._trial["trial_id"])]
        horizontal, spatial = distance_to_target(position, targets)
        success = bool(horizontal <= SUCCESS_DISTANCE_M)
        if self._attempts and self._attempts[-1]["step"] == int(self._frame_id):
            # The episode loop asks before stepping and habitat's terminal STOP
            # follows at the same pose: one stop, not two.
            return bool(self._attempts[-1]["success"])
        self._attempts.append(
            {
                "attempt": len(self._attempts) + 1,
                "step": int(self._frame_id),
                "position": [float(x) for x in position],
                "distance_horizontal_m": horizontal,
                "distance_3d_m": spatial,
                "success": success,
            }
        )
        return success

    def attempt_scored(self, frame, cfg) -> bool:
        """Would this STOP answer the query? Asked without ending the episode."""
        return self._record_stop()

    def episode_metadata(self) -> Dict[str, Any]:
        """What the record needs that only this env knows: the protocol result."""
        meta = dict(self._trial)
        position = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
        horizontal, spatial = distance_to_target(
            position, self._targets[str(self._trial["trial_id"])]
        )
        attempts = list(self._attempts)
        # Success is earned at a STOP and nowhere else. An episode that runs out
        # of steps standing next to the object has not answered the query --
        # which is exactly how the released harness scores DualMap, where the
        # keyframe budget expiring is a failure however close the agent is.
        success = any(bool(a["success"]) for a in attempts[:MAX_ATTEMPTS])
        travelled = float(self._travelled)
        spl = 0.0
        if success and math.isfinite(self._shortest):
            spl = self._shortest / max(self._shortest, travelled, 1e-12)
        meta["dualmap"] = {
            "trial_id": str(self._trial.get("trial_id")),
            "condition": str(self._trial.get("condition")),
            "query": str(self._trial.get("query")),
            "agent_query": str(self._trial.get("agent_query")),
            "success": int(success),
            "spl": float(spl),
            "travelled_m": travelled,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "final_distance_horizontal_m": horizontal,
            "final_distance_3d_m": spatial,
            "shortest_success_path_m": (
                None if not math.isfinite(self._shortest) else float(self._shortest)
            ),
            "reference_shortest_success_path_m": self._trial.get(
                "reference_shortest_success_path_m"
            ),
            "start": dict(self._start_snap),
            "success_definition": (
                f"horizontal distance to any queried instance <= "
                f"{SUCCESS_DISTANCE_M} m, within {MAX_ATTEMPTS} attempts"
            ),
        }
        return meta

    def metrics(self) -> dict:
        """Report the benchmark's metric, not habitat's.

        Habitat scores this episode by ITS success distance against goal view
        points, and that number would otherwise land in `episodes.jsonl`,
        `summary.json` and the console line as if it were the comparison. It is
        kept, under its own name, because a run's own criterion is worth having
        beside the one it is being compared under -- but `success` and `spl`
        here mean what the released benchmark means by them.
        """
        base = dict(self.env.get_metrics())
        block = self.episode_metadata()["dualmap"]
        base["habitat_success"] = float(base.get("success", 0.0))
        base["habitat_spl"] = float(base.get("spl", 0.0))
        base["habitat_distance_to_goal"] = float(base.get("distance_to_goal", -1.0))
        base["success"] = float(block["success"])
        base["spl"] = float(block["spl"])
        base["distance_to_goal"] = float(block["final_distance_horizontal_m"])
        return base

    def benchmark_metadata(self) -> Dict[str, Any]:
        return {
            "mode": "dualmap_protocol",
            "release_root": str(RELEASE_ROOT),
            "reference_run": str(self.cfg.dualmap.reference_run),
            "success_distance_m": SUCCESS_DISTANCE_M,
            "max_attempts": MAX_ATTEMPTS,
            "trials": len(self.trials),
            "conditions": sorted({t["condition"] for t in self.trials}),
            "scenes": sorted({t["scene"] for t in self.trials}),
        }
