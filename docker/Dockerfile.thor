# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# The pipeline and the perception models, on a Jetson AGX Thor.
#
# This is the Jetson half of the deployment: arm64, NVIDIA's own CUDA and
# PyTorch, and NOTHING ELSE from the x86 image. Two things it deliberately
# does not contain:
#
#   HABITAT. There is no aarch64 build of habitat-sim -- the `aihabitat` conda
#   channel is linux-64 only -- and the robot does not need one. Verified: the
#   whole robot path (Ros2Env -> NavAgent -> run_episode -> save_map) runs with
#   habitat blocked at import. `eval.mode=ros2` is the only mode this image can
#   run; the benchmark modes stay on the x86 image.
#
#   ROS. The bridge lives in its own container (docker/Dockerfile.bridge),
#   because `src/osg/ros2/` was written to import under two interpreters and a
#   bridge that needs only numpy + rclpy has no business carrying CUDA. That
#   split is also what lets the bridge match the robot's Ubuntu 22.04 / Humble
#   exactly while this image stays on the JetPack-side Ubuntu -- no ROS built
#   from source, no distro gap to bridge, no Python-version fight with torch.
#
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env build
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env up -d
#
# See docs/THOR.md. Every ARG below is a "verify on YOUR unit" item.
# ---------------------------------------------------------------------------

# The base image, which must match the JetPack ON THE DEVICE. Check it with
#     cat /etc/nv_tegra_release
#
# NOT an `l4t-*` tag, and this is the correction that made the build work at
# all. The Jetson-specific container line stops at JetPack 6 / Orin:
# `l4t-jetpack` publishes nothing past r36.4.0, `l4t-cuda` nothing past 12.6,
# so the r38/r39 tags this file used to name do not exist and the build died
# on `no such manifest` before running a single instruction. JetPack 7 (this
# unit reports L4T R39.2) dropped that split -- the ordinary arm64 NGC images
# run on the device. `nvidia/pytorch:25.10-py3` is Ubuntu 24.04 + CUDA 13.0 +
# torch 2.9 compiled with sm_110 among its architectures, which is exactly
# what the GPU here reports (compute capability 11.0), and it ships nvcc, so
# GroundingDINO's fused attention kernel still compiles in-image.
#
# It also removes a step: torch is IN the base, so no Jetson wheel index is
# needed -- which matters, because the one this file used to install from
# (pypi.jetson-ai-lab.dev) now answers `410 Gone - permanently removed`.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Thor's GPU is Blackwell-generation and reports 11.0; Orin is 8.7. Check with
#     python3 -c "import torch; print(torch.cuda.get_device_capability())"
# and rebuild with the right value if this is wrong -- a kernel compiled for
# the wrong arch falls back to CPU, which for GroundingDINO is the difference
# between a run and a day.
ARG CUDA_ARCH=11.0
# Empty, and normally left empty: the base image carries torch and torchvision.
# It exists for a JetPack-6 base (`l4t-jetpack:r36.4.0`), which does not --
# point it at NVIDIA's Jetson wheel index for that JetPack and torch is
# installed from there instead. Do not point it at a JetPack-7 index; there
# isn't one.
ARG TORCH_INDEX_URL=
ARG TORCH_SPEC=torch
ARG TORCHVISION_SPEC=torchvision
# BLIP-2 arrives through salesforce-lavis, which has no Jetson wheels and pins
# an old transformers. It feeds only the value map and ASCENT's commit gate,
# and `exploration.value_model=clip` replaces it -- so it is optional rather
# than allowed to block the whole image.
ARG WITH_BLIP2=1

# Only what the base does not already have. git, curl, the compilers, cmake,
# ninja and the Python headers are in the NGC image; ffmpeg, the GL/glib
# libraries OpenCV links against, and sudo (entrypoint.thor.sh chowns the
# mounted runtime dirs with it) are not. They are all still listed, because
# apt installs into /usr/bin and the base's own copies live in /usr/local,
# which comes first on PATH -- so this list is also what makes an l4t base
# work. python3-pip and python3-venv are the exception and are deliberately
# ABSENT: the base's pip is a /usr/local install against the system python,
# and apt's would put a second, older one in /usr/bin.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
       git curl ca-certificates build-essential ninja-build cmake pkg-config \
       python3-dev \
       libjpeg-dev libpng-dev libopenblas-dev libgl1 libglib2.0-0 \
       sudo tmux ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

# ---------------------------------------------------------------------------
# 1. The pipeline environment: the base image's python, and its torch.
# ---------------------------------------------------------------------------
# `numpy>=2` -- the OPPOSITE of the x86 image's `numpy<2`, for the same reason:
# match the ABI the torch in use was built against. The x86 wheel is a 1.x
# build; the arm64 torch here is built against numpy 2.1, and a 1.x numpy
# under it is what produces `_ARRAY_API not found`. A constraint is still
# needed rather than nothing, because several packages below (and lavis, at
# the end) will otherwise pull 1.x in as a dependency.
#
# It is written into /etc/pip/constraint.txt because the NGC base already
# points PIP_CONSTRAINT there and ships the file empty, for exactly this. The
# mkdir and the ENV are what make the same line work on an l4t base, which
# has neither. Unlike the x86 image this is NOT cleared for run time: a later
# `pip install` inside the container has the same ABI to respect.
#
# `opencv-python*<5` is in the same file because it has to hold in BOTH
# environments and nothing here asks for OpenCV by name: ultralytics pulls
# `opencv-python` in, lavis pulls `opencv-python-headless`, and left to float
# both now resolve to 5.0 -- a major release the recorded ASCENT environment
# has never run (it is on 4.10.0.84). A constraint rather than a requirement,
# so it binds whoever drags OpenCV in.
RUN mkdir -p /etc/pip \
 && printf 'numpy>=2\nopencv-python<5\nopencv-python-headless<5\n' \
      >> /etc/pip/constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        pip3 install --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple \
            ${TORCH_SPEC} ${TORCHVISION_SPEC}; \
    fi \
 && python3 - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    # Not fatal at BUILD time -- the builder may have no GPU visible -- but the
    # message has to name the two knobs that actually cause it.
    print("WARNING: torch.cuda unavailable during build. If this persists at "
          "run time, BASE_IMAGE does not match the device's JetPack.")
PY

# The pipeline's own dependencies, from pyproject. Installed by name rather
# than `pip install -e .` so a source checkout is not required in the image:
# the repo is bind-mounted and reached through PYTHONPATH, exactly as on x86.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        numpy scipy scikit-learn scikit-image \
        opencv-python-headless "hydra-core>=1.3" omegaconf "openai>=1.30" \
        imageio matplotlib pillow pyyaml tqdm requests \
        "ultralytics==8.3.*" pytest pytest-timeout

# region_proposal.enabled -- on in every unified preset, and inherited by the
# Stretch ones -- encodes crops with MobileCLIP-S2 through open_clip, so a run
# without these dies in build_run_components with `No module named
# 'open_clip'` before the robot has moved. --no-deps on ml-mobileclip is
# deliberate and is what docs/SETUP.md prescribes: it supplies only
# `reparameterize_model`, and its dependency pins would otherwise move torch.
# The MobileCLIP-S2 weights themselves resolve from the HF cache at run time.
#
# ultralytics' CLIP fork is the detector's text encoder: YOLOE's `set_classes`
# (which is how the target vocabulary reaches the detector, and how
# `mobileclip_blt.ts` gets staged) imports `clip`, and ultralytics' own
# on-demand install of it does not fire under a non-root user. Same line the
# x86 image carries (docker/Dockerfile).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install open_clip_torch \
 && pip3 install --no-deps "git+https://github.com/apple/ml-mobileclip.git" \
 && pip3 install "git+https://github.com/ultralytics/CLIP.git" \
 && python3 -c "import open_clip, mobileclip, clip; print('open_clip', open_clip.__version__)"

# ---------------------------------------------------------------------------
# 2. The five perception model servers.
# ---------------------------------------------------------------------------
# A venv WITH system site-packages, so torch is shared rather than installed a
# second time (several GB on this platform), while lavis's old transformers and
# whatever else it drags along land in the venv layer and shadow the
# pipeline's rather than overwriting it. Same separation the x86 image gets
# from its second conda env, for the same reason -- and `serve_perception.sh`
# already selects it through ASCENT_PYTHON, so that script needs no changes.
ENV ASCENT_PYTHON=/opt/venvs/models/bin/python
RUN python3 -m venv --system-site-packages /opt/venvs/models

# The reference source. `relative_work/ascent` is a git submodule, so the build
# FAILS HERE unless it was checked out recursively first:
#     git submodule update --init --recursive
# Recursively matters: `recognize-anything`, `places365` and `vlfm` are
# submodules OF that submodule, and a plain `--init` leaves them empty.
COPY relative_work/ascent /opt/ascent

COPY docker/thor-model-requirements.txt /opt/thor-model-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ASCENT_PYTHON} -m pip install --upgrade pip setuptools wheel \
 && ${ASCENT_PYTHON} -m pip install -r /opt/thor-model-requirements.txt

WORKDIR /opt/ascent
# The editable installs the servers import. --no-deps throughout: the resolver
# is what pulls a numpy over the constraint and breaks torch underneath.
RUN for pkg in third_party/MobileSAM \
               third_party/frontier_exploration \
               third_party/depth_camera_filtering; do \
        ${ASCENT_PYTHON} -m pip install --no-cache-dir --no-deps -e "$pkg" || exit 1; \
    done

# GroundingDINO's MultiScaleDeformableAttention kernel is older than the torch
# it is now compiled against. Its two `AT_DISPATCH_FLOATING_TYPES(value.type()`
# calls pass a `DeprecatedTypeProperties` where a `ScalarType` is wanted; the
# overload that used to accept it was deprecated in torch 1.x and is gone in
# 2.4+, so nvcc stops with `no suitable conversion function from "const
# at::DeprecatedTypeProperties" to "c10::ScalarType"`. The x86 image never
# sees this because it pins torch 2.1, where the overload still exists.
#
# Patched HERE rather than in the checkout: `relative_work/ascent` is a
# submodule pinned to an upstream commit, and editing it would make every
# `git submodule status` dirty for a change that only this platform needs.
# The count is echoed so an upstream fix shows up as "0" rather than silently
# turning this into a no-op sed.
RUN set -eu; \
    f=/opt/ascent/third_party/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu; \
    n="$(grep -c 'AT_DISPATCH_FLOATING_TYPES(value\.type()' "$f" || true)"; \
    if [ "$n" != "0" ]; then \
        sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$f"; \
        echo "ms_deform_attn_cuda.cu: patched $n AT_DISPATCH call(s) for torch>=2.4"; \
    else \
        echo "ms_deform_attn_cuda.cu: nothing to patch (upstream fixed it)"; \
    fi

# GroundingDINO: compile the fused attention kernel against THIS platform's
# nvcc. `pip install -e .` is not used -- setuptools' develop re-invokes pip in
# an isolated env with no torch, and GroundingDINO's setup.py responds by
# trying to install torch from inside it. A .pth makes it importable instead,
# which is how the x86 image carries it too.
#
# The .pth path is asked of the interpreter rather than globbed. A glob in a
# REDIRECTION TARGET only expands against files that already exist, and the
# .pth is the file being created -- so `> .../python3*/site-packages/x.pth`
# leaves the `*` literal and fails with `No such file or directory`, however
# right the directory underneath it is.
RUN cd /opt/ascent/third_party/GroundingDINO \
 && ${ASCENT_PYTHON} setup.py build_ext --inplace \
 && echo /opt/ascent/third_party/GroundingDINO \
      > "$(${ASCENT_PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_ascent_gdino.pth" \
 && ${ASCENT_PYTHON} -c "import torch; import groundingdino; \
      from groundingdino import _C; print('GroundingDINO kernel OK')"

RUN echo /opt/ascent/third_party/recognize-anything \
      > "$(${ASCENT_PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_ascent_ram.pth" \
 && ${ASCENT_PYTHON} -c "import ram; print('recognize-anything OK')"

# BLIP-2 last, and optional: see WITH_BLIP2 above.
#
# --no-deps, with the dependency list rebuilt by hand, because ONE of lavis's
# declared dependencies cannot be satisfied on this architecture at all:
#
#   decord has no aarch64 distribution -- no wheel, and no version that builds
#   -- so a plain `pip install salesforce-lavis` ends in `ResolutionImpossible`
#   naming it. `decord2` is a maintained fork that publishes manylinux aarch64
#   wheels AND installs its module as `decord`, which is the name lavis
#   imports (`lavis.datasets.data_utils`, reached from `lavis/__init__.py`).
#   That is the whole of the fix: with it, `import decord; import lavis` both
#   succeed here.
#
# lavis's exact pins are dropped rather than honoured, which is also what the
# x86 environment does -- `docker/ascent-requirements.txt`, a freeze of a
# working ASCENT install, records opencv 4.10.0.84 against lavis's ==4.5.5.64
# and fairscale 0.4.13 against its ==0.4.4. Those, plus timm and transformers,
# come from thor-model-requirements.txt and so are absent from the list below.
# streamlit, pre-commit and ipython are its demo and dev extras; `import lavis`
# does not reach them.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WITH_BLIP2}" = "1" ]; then \
        ${ASCENT_PYTHON} -m pip install --no-cache-dir decord2 \
          && ${ASCENT_PYTHON} -m pip install --no-cache-dir --no-deps salesforce-lavis \
          && ${ASCENT_PYTHON} -m pip install --no-cache-dir \
               contexttimer iopath opendatasets pandas plotly pycocoevalcap \
               python-magic scikit-image spacy webdataset \
          && ${ASCENT_PYTHON} -c "import decord; import lavis; print('lavis OK')" \
          || { echo "BLIP-2 did not install on aarch64. Rebuild with"; \
               echo "  --build-arg WITH_BLIP2=0   (or WITH_BLIP2=0 in docker/.env)"; \
               echo "and run with exploration.value_model=clip."; exit 1; }; \
    else \
        echo "BLIP-2 skipped (WITH_BLIP2=0): use exploration.value_model=clip"; \
    fi

# What the servers actually ended up with. Everything above installs by name,
# and every one of these is also reachable as somebody else's transitive
# dependency -- this is the step that says so out loud, rather than letting a
# silent upgrade be discovered by a model server at run time, on the robot.
# The wanted versions are the recorded ASCENT environment's.
RUN ${ASCENT_PYTHON} - <<'VERSIONS'
import cv2, numpy, timm, transformers
got = {"transformers": transformers.__version__, "timm": timm.__version__,
       "cv2": cv2.__version__, "numpy": numpy.__version__}
want = {"transformers": "4.37.", "timm": "0.4.", "cv2": "4.", "numpy": "2."}
bad = [f"{k}={v} (wanted {want[k]}x)" for k, v in got.items()
       if not v.startswith(want[k])]
if bad:
    raise SystemExit("models venv drifted: " + "; ".join(bad))
print("models venv:", ", ".join(f"{k} {v}" for k, v in sorted(got.items())))
VERSIONS

# ---------------------------------------------------------------------------
# 3. Runtime
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/workspace/src \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    OPENBLAS_NUM_THREADS=4

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
# Ubuntu 24.04 -- which the NGC base is -- SHIPS a `ubuntu` user at uid 1000,
# so the old unconditional `useradd ... || true` silently did nothing on the
# common UID and the image then failed at run time with `unable to find user
# user`. Take the id over instead, and only create what is missing.
RUN set -eu; \
    if getent passwd $USER_UID >/dev/null; then \
        old="$(getent passwd $USER_UID | cut -d: -f1)"; \
        if [ "$old" != "$USERNAME" ]; then \
            userdel -r "$old" >/dev/null 2>&1 || userdel "$old" >/dev/null 2>&1 || true; \
        fi; \
    fi; \
    getent group $USER_GID >/dev/null || groupadd -g $USER_GID $USERNAME; \
    getent passwd $USERNAME >/dev/null \
        || useradd -m -s /bin/bash -u $USER_UID -g $USER_GID $USERNAME; \
    usermod -aG sudo $USERNAME; \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers; \
    chown -R $USER_UID:$USER_GID /opt/ascent /opt/venvs

# The bind mount is shared with the host user; without this every file written
# from a `docker exec` shell comes out 0644 and the host cannot modify it.
RUN printf 'umask 002\n' > /etc/profile.d/00-shared-umask.sh \
 && printf '\numask 002\n' >> /etc/bash.bashrc

COPY docker/entrypoint.thor.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $USERNAME
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
