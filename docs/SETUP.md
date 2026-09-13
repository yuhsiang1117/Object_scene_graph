# Setup: the repo, the submodule, and the model weights

Everything below runs inside the `nav` container (`README.md` → Quick start).
Two things this repo does **not** contain and cannot run without: the ASCENT
reference checkout (a pinned submodule) and ~5 GB of model weights (downloaded).

---

## 1. Clone with the submodule

`relative_work/ascent` is a submodule pinned to
[Zeying-Gong/ascent](https://github.com/Zeying-Gong/ascent) at `8f7bbf9`. It is
where the five perception models live, so nothing runs without it.

```bash
git clone --recursive <this repo>
# already cloned without --recursive:
git submodule update --init --recursive
```

**`--recursive` is not optional, and the image build needs it first.** The
Dockerfile does `COPY relative_work/ascent /opt/ascent`, so the build fails
outright if the submodule is not checked out. Running from the checkout needs
it too: without it ASCENT's nine nested submodules are empty directories and
the servers import nothing.

The build context is the repo root (`context: ..`), which `/.dockerignore`
trims from 35 GB on disk to ~213 MB — it drops `.conda-envs/`, `data/`,
`outputs/`, the 4.5 GB of model weights, and `docker/.env`.
`tests/unit/test_docker_build_context.py` checks that no `COPY` names a path
the ignore file excludes, which fails a build as confusingly as a typo.

ASCENT itself carries nine nested submodules (GroundingDINO, MobileSAM,
D-FINE, RAM++, places365, vlfm, frontier_exploration, depth_camera_filtering,
habitat-lab), and a non-recursive clone leaves all nine as empty directories.
The servers import from them.

### The compiled CUDA extension (only for pre-merge containers)

The merged image builds it at image-build time, so a container from
`docker/Dockerfile` already has it. The rest of this section applies only if
you are on an older container or building the extension by hand.


GroundingDINO ships a CUDA op (`MultiScaleDeformableAttention`) that must be
**built in the working tree**. It is not in git, so a fresh clone does not have
it, and the failure is not obvious — the server starts, answers its health
check, and then returns HTTP 500 on the first real frame with

```
NameError: name '_C' is not defined
```

which the run surfaces as `PerceptionUnavailable: …/gdino: HTTP Error 500`.
Build it once, in the `ascent` env — **compile in place and add a `.pth`; do
not use `pip install -e .`**, which re-enters pep517 in an isolated environment
and makes GroundingDINO's `setup.py` try to `pip install torch` from inside it:

```bash
ASCENT_PY=/workspace/.conda-envs/ascent/bin/python
cd relative_work/ascent/third_party/GroundingDINO
CUDA_HOME=/usr/local/cuda $ASCENT_PY setup.py build_ext --inplace
$ASCENT_PY -c "import site, pathlib; \
  pathlib.Path(site.getsitepackages()[0], '_ascent_gdino.pth') \
  .write_text('$PWD\n')"
cd - && $ASCENT_PY -c "import torch; from groundingdino import _C; print('OK')"
```

It produces `groundingdino/_C.cpython-39-*.so` in that directory; check the
file exists before blaming the model. Import torch first when checking — the
extension links `libc10` with no RPATH, so a bare `from groundingdino import
_C` fails even on a good install.

---

## 2. `docker/.env`

`docker/.env.example` lists every variable `compose.yaml` reads. Two must be
set by hand:

```bash
cp docker/.env.example docker/.env
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> docker/.env
```

- **`UID` / `GID`** — the image builds its non-root user with these, and
  `/workspace` is a bind mount of your checkout. Get them wrong and the
  container cannot write `outputs/`. They cannot be inherited from your shell:
  `UID` is a bash built-in that is **not exported**, so compose never sees it
  and silently falls back to 1000.
- **`HM3D_SCENES`** — where the license-gated scene meshes live, mounted onto
  `data/versioned_data`. Wrong or unset, the eval does not report a missing
  file; it dies in the env worker with `No Stage Attributes exists for
  requested scene ...basis.glb`.

`NVIDIA_API_KEY` is only needed for `verification=nim`, which is off in the
default arm. The collector paths only matter for the `ycb_*` experiments.

## 3. Model weights

Nothing is committed. `pretrained_weights/` inside the submodule is gitignored,
and so is `data/`.

### The five served models + the mover

```bash
bash scripts/fetch_ascent_weights.sh        # -> relative_work/ascent/pretrained_weights/
```

| file | size | used by | source |
|---|---|---|---|
| `groundingdino_swint_ogc.pth` | 662 MB | stair detector (:13184) | GroundingDINO v0.1.0-alpha release |
| `ram_plus_swin_large_14m.pth` | 2.9 GB | RAM++ tagger (:13185) | HF `xinyu1205/recognize-anything-plus-model` |
| `dfine_x_obj2coco.pth` | 240 MB | D-FINE detector (:13186) | Peterande/storage dfinev1.0 |
| `mobile_sam.pt` | 39 MB | mask per detection (:13183) | linked from `data/weights/` |
| `rednet_semmap_mp3d_40.pth` | 626 MB | stair segmentation | linked from `data/weights/` |
| `resnet50_places365.pth.tar` | 93 MB | room labels (in-process) | linked from `data/place365/` |
| `pointnav_weights.pth` | 32 MB | the frozen mover | linked from `data/weights/` |

The script links the four this repo already has and downloads the other three;
run it again any time — it skips what is present. **BLIP-2** (the value map and
commit gate, :13182) is not a file here: `lavis` pulls it from HF on the
server's first start, which is why that server takes longest to come up.

It also offers Qwen2.5-7B-Instruct (~15 GB) for ASCENT's own LLM server. **You
do not need it**: both this repo and the reference call `qwen2.5:7b` on the
local ollama instead. Skip it unless you are running native ASCENT with its
bf16 server.

### The OSG-side weights

```bash
python scripts/download_weights.py                 # detector + mobileclip
python scripts/download_weights.py --pointnav      # the mover
python scripts/download_weights.py --rednet        # stair segmentation
python scripts/download_weights.py --clip          # value-map image-text model
```

### The LLM

```bash
docker exec docker-ollama-1 ollama pull qwen2.5:7b     # 4.7 GB, Q4_K_M
```

Note it is **quantised** — ASCENT's paper configuration is bf16. Both sides
here use the same quantised model, so comparisons between them are fair; see
`docs/METHODOLOGY.md` §5.4.

### Datasets

```bash
python scripts/download_data.py --username <TOKEN_ID> --password <TOKEN_SECRET> \
    --uids hm3d_val_v0.2
python scripts/download_data.py --episodes-only
```

HM3D scenes need a free licence from
https://matterport.com/habitat-matterport-3d-research-dataset.

---

## 4. Check it works

```bash
bash scripts/serve_perception.sh                     # five servers, ~60 s to load
python scripts/run_eval.py eval.num_episodes=1       # one live episode
```

A green run prints one `[1/1] … success=…` line. If it raises
`PerceptionUnavailable`, the named port tells you which model is unhappy;
`relative_work/ascent/debug/vlm_logs/<name>.log` has the server-side traceback.

Total on disk: ~5 GB of weights, ~1 GB of nested submodule source, plus HM3D
scenes (~10 GB for val).

---

## 5. The two conda environments

One image, two environments — `docker/Dockerfile` builds both:

| env | python | torch | for |
|---|---|---|---|
| `habitat` (default on shell entry) | 3.9 | 2.4.1+cu121 | habitat-sim, OSG, YOLOE, the eval harness |
| `ascent` | 3.9 | 2.1.0+cu118 | BLIP-2, MobileSAM, GroundingDINO, RAM++, D-FINE, and the native reference |

They cannot be one interpreter: habitat-sim 0.3.1 pins numpy < 1.24, lavis
needs a transformers the OSG side does not want, and the torch builds differ.
That is exactly why the models are served over HTTP rather than imported.

`scripts/serve_perception.sh` finds the ascent env at
`/opt/conda/envs/ascent`, falling back to a workspace-local
`.conda-envs/ascent` for containers built before the two Dockerfiles were
merged. Override with `ASCENT_PYTHON=…`.

Verify both are intact:

```bash
python -c "import habitat_sim, torch, ultralytics; print('habitat env OK')"
$ASCENT_PYTHON -c "
import habitat_sim, lavis, mobile_sam, groundingdino.util.inference, ram
from groundingdino import _C
print('ascent env OK')"
```

The base image is `nvidia/cuda:11.8.0-cudnn8-devel`. GroundingDINO's kernel
must be compiled, torch refuses to build an extension when the toolchain's
version differs from `torch.version.cuda`, and a devel base puts a matching
11.8 nvcc at `/usr/local/cuda` where torch looks for it. The build asserts the
two agree before compiling, so a mismatch fails in seconds rather than
producing a silent CPU fallback that costs 14-40 hours on a 100-episode run.

The pip half of that env is pinned in `docker/ascent-requirements.txt`,
generated from its `pip freeze` and installed with `--no-deps`. The list is a
complete closure, so re-running the resolver is unnecessary — and actively
harmful, since habitat-lab's unbounded `numpy>=1.20.0` otherwise pulls numpy 2
over the pin and breaks `import torch`. GroundingDINO and RAM++ are wired by
`.pth`, matching `_ascent_gdino.pth` / `_ascent_ram.pth` in the live env.

## 6. What the submodule does and does not carry

The submodule is pinned to **upstream, unmodified**. During this work the
checkout carried three local patches which are deliberately no longer applied:

| patch | what it did | consequence of dropping it |
|---|---|---|
| behaviour recorder (+94 lines in `ascent_policy.py`, `ascent_trainer.py`, `llm_planner.py`, plus `behaviour_recorder.py`) | produced ASCENT's per-step trace | the reference trace cannot be regenerated. The trace **itself is preserved** in this repo at `data/reference/ascent_behaviour_100/` (gzipped, 772 KB), which is what `scripts/compare_ascent_osg.py` reads by default |
| a lock in `model_api/server_wrapper_out.py` | serialised model calls | the servers are **not reentrant** again. GroundingDINO caches per-image state on the module, so two concurrent evals race and the loser gets HTTP 500. **Run one evaluation at a time.** This repo's client retries once, which covers a transient hit, not sustained concurrency |
| `model_api/qwen25_ollama.py` | served ASCENT's Qwen endpoint from ollama | only matters when running native ASCENT; its own launcher expects the bf16 server |

Nothing in this repo's own pipeline depends on those patches — verified by a
live episode against the clean submodule after the conversion.
