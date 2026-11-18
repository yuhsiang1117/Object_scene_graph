# One image, every experiment: OSG's agent, the ASCENT reference, and all five
# perception servers.
#
#   git submodule update --init --recursive        # REQUIRED: the build COPYs it
#   docker compose -f docker/compose.yaml --env-file docker/.env build nav
#
# The build context is the REPO ROOT (compose sets `context: ..`), kept to
# ~264 MB by /.dockerignore rather than the 35 GB on disk.
#
# TWO CONDA ENVS, ONE IMAGE. They cannot be merged into one interpreter and
# every attempt is encoded below:
#
#   habitat  python 3.9, torch 2.4.1+cu121, numpy<2   -- habitat-sim, OSG, YOLOE
#   ascent   python 3.9, torch 2.1.0+cu118, numpy 1.23.5, transformers 4.37.0
#            -- BLIP-2 (lavis), MobileSAM, GroundingDINO, RAM++, D-FINE
#
# habitat-sim 0.3.1 pins numpy<1.24; lavis needs a transformers that the OSG
# side does not want; and the two torch builds differ. Process-per-model over
# HTTP is ASCENT's own architecture, and the split env is why it is necessary.
#
# THE BASE IS `11.8.0-devel`, and that is the one thing here established by
# building rather than by reading: GroundingDINO's fused attention kernel
# compiled cleanly on it, twice, before unrelated steps failed.
#
# torch's `_find_cuda_home` takes $CUDA_HOME, else the directory above whatever
# `which nvcc` returns, and then refuses to build when that toolchain's version
# differs from `torch.version.cuda` (11.8 here). A `-runtime-` base has no nvcc
# at all, so the toolchain has to come from somewhere; a devel base puts a
# matching 11.8 at /usr/local/cuda and needs no further arrangement.
#
# The conda route was tried and reverted. The live env does carry conda
# `cuda-toolkit=11.8`, which is what led me to think it had compiled there --
# but its only nvcc is now 13.3 (a later package clobbered the 11.8 one) and
# torch still resolves CUDA_HOME to /usr/local/cuda. That env's `_C.so` was
# almost certainly compiled in a devel-based container and carried across, so
# it is evidence for the devel base, not against it.
#
# The habitat env's torch 2.4.1+cu121 is unaffected: torch wheels bundle their
# own CUDA runtime and nothing on the OSG side compiles an extension.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# EGL/GL userland for headless habitat-sim (driver libs are injected by
# nvidia-container-toolkit at run time); build-essential/ninja for the CUDA
# extension; tmux because the model servers run in a tmux session. ffmpeg and
# xvfb are also used by habitat-data-collector's recording and headless
# authoring scripts.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 unzip build-essential ninja-build \
    libegl1 libgl1 libglvnd0 libx11-6 libxext6 libopengl0 libglx0 \
    libjpeg-dev libpng-dev libglib2.0-0 sudo tmux ffmpeg xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# ---------------------------------------------------------------------------
# 1. `habitat`: the OSG side. habitat-sim 0.3.1 is a python 3.9 build.
# ---------------------------------------------------------------------------
RUN conda create -y -n habitat python=3.9 cmake=3.14.0 \
    && conda install -y -n habitat habitat-sim=0.3.1 withbullet headless \
       -c conda-forge -c aihabitat \
    && conda clean -ya

SHELL ["conda", "run", "--no-capture-output", "-n", "habitat", "/bin/bash", "-c"]

RUN git clone --depth 1 --branch v0.3.1 \
      https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
    && pip install --no-cache-dir -e /opt/habitat-lab/habitat-lab

RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
       "numpy<2" "ultralytics==8.3.*" openai "hydra-core>=1.3" omegaconf \
       scipy scikit-image scikit-learn "opencv-python==4.10.0.84" \
       pytest pytest-timeout imageio matplotlib pillow pyyaml tqdm anthropic \
       "rerun-sdk"

# habitat-data-collector shares this interpreter. habitat-sim is already
# supplied above (the compatible 0.3.1 Conda build), while its remaining
# non-ROS requirements are included in the install above. ROS 2 is optional in
# the collector and is intentionally not mixed into this Python 3.9 env:
# Ubuntu 22.04's ROS Humble binaries, including rclpy, target Python 3.10.
#
# The Stretch 3 layer lives with that rather than fighting it. Humble IS
# installed further down (install_ros2_humble.sh), against the system
# python3.10, and the pipeline reaches it over a local socket instead of an
# import -- `src/osg/ros2/` is written to run under both interpreters. Nothing
# sources setup.bash into a login shell for the same reason this comment
# exists: ROS's 3.10 site-packages must never precede this env's imports.

# CLIP tokenizer for YOLOE's text prompts (otherwise ultralytics AutoUpdate
# re-installs it in every ephemeral container).
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# ---------------------------------------------------------------------------
# 2. `ascent`: the five perception servers and the native reference.
#
# ASCENT's own README is the spine (python 3.9, torch 2.1.0+cu118,
# habitat-sim 0.3.1, transformers==4.37.0 installed LAST -- that override of
# lavis's `<4.27` pin is ASCENT's, and it is what lets BLIP-2 and Qwen2.5
# share one environment).
# ---------------------------------------------------------------------------
# No conda CUDA here: the base image's 11.8 toolchain at /usr/local/cuda is
# what torch's cpp_extension finds, and adding a second one only creates a
# chance for the wrong nvcc to win.
RUN conda create -y -n ascent python=3.9 cmake=3.14.0 && conda clean -ya
SHELL ["conda", "run", "--no-capture-output", "-n", "ascent", "/bin/bash", "-c"]

RUN conda install -y -n ascent habitat-sim=0.3.1 withbullet headless \
      -c conda-forge -c aihabitat && conda clean -ya

# Everything pip-provided, pinned exactly as it resolved in the environment
# that runs the servers (docker/ascent-requirements.txt is generated from its
# `pip freeze`; see its header).
#
# --no-deps, deliberately. The list is already a complete, consistent closure,
# and re-running the resolver is what broke the previous attempts: habitat-lab
# asks for `numpy>=1.20.0`, which pulls numpy 2.x over the pin and makes
# `import torch` fail with "_ARRAY_API not found". It also sidesteps
# salesforce-lavis, whose declared deps cannot resolve on python 3.9 at all.
# PIP_CONSTRAINT holds numpy for anything installed later in this env.
RUN printf 'numpy==1.23.5\n' > /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt
COPY docker/ascent-requirements.txt /opt/ascent-requirements.txt
RUN pip install --no-cache-dir --no-deps -r /opt/ascent-requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
    && python -c "import torch, numpy, transformers, cv2; \
       assert numpy.__version__.startswith('1.23'), numpy.__version__; \
       print('ascent pins:', torch.__version__, numpy.__version__, transformers.__version__)"

# The reference source. `relative_work/ascent` is a git submodule, so the build
# FAILS HERE unless it was checked out recursively first:
#
#     git submodule update --init --recursive
#
# It is COPYed rather than cloned so the commit that ships is whatever git
# records -- no second copy of the pin to drift, and local changes to the
# reference are reflected.
COPY relative_work/ascent /opt/ascent
WORKDIR /opt/ascent

# The six editable installs the live env carries (read off its
# direct_url.json). frontier_exploration and depth_camera_filtering were
# missing from the old hand-written recipe; ASCENT's policy and vlfm import
# both. --no-deps throughout: the closure above is already complete.
RUN for pkg in third_party/habitat-lab/habitat-lab \
               third_party/habitat-lab/habitat-baselines \
               third_party/MobileSAM third_party/vlfm \
               third_party/frontier_exploration \
               third_party/depth_camera_filtering; do \
        pip install --no-cache-dir --no-deps -e "$pkg" || exit 1; \
    done \
    && python -c "import numpy; assert numpy.__version__.startswith('1.23'), numpy.__version__; \
       print('numpy held at', numpy.__version__)"

# GroundingDINO: compile the fused attention kernel, then make the package
# importable with a .pth -- which is exactly how the live env has it
# (_ascent_gdino.pth). `pip install -e .` is NOT used: setuptools' `develop`
# re-invokes `pip install -e . --use-pep517` in an isolated environment with
# neither torch nor pip, and GroundingDINO's setup.py responds to the failed
# torch import by trying to `pip install torch` from inside it.
#
# CUDA_HOME is the base image's toolchain, which matches torch's cu118.
# Without a working kernel GroundingDINO falls back to CPU -- at one call per
# step that is 14-40 hours for a 100-episode run.
#
# TORCH MUST BE IMPORTED FIRST in the check. _C.so links libc10 with no RPATH,
# so it resolves only once torch has loaded libc10.so; a bare
# `from groundingdino import _C` fails with "libc10.so: cannot open shared
# object file" even on a perfectly good install (confirmed against the live
# env). The servers are unaffected -- groundingdino/util/inference.py imports
# torch at its top.
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
# Fail here, loudly, rather than 40 hours into a run on a CPU fallback.
RUN nvcc --version | tail -2 \
    && python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; \
       import subprocess, re; \
       v = subprocess.check_output([CUDA_HOME + '/bin/nvcc', '--version']).decode(); \
       n = re.search(r'release ([0-9.]+)', v).group(1); \
       assert n == torch.version.cuda, f'nvcc {n} != torch cuda {torch.version.cuda}'; \
       print('toolchain matched: nvcc', n, '== torch', torch.version.cuda)"
RUN cd third_party/GroundingDINO \
    && python setup.py build_ext --inplace \
    && python -c "import site, pathlib; \
       pathlib.Path(site.getsitepackages()[0], '_ascent_gdino.pth') \
       .write_text('/opt/ascent/third_party/GroundingDINO\n')" \
    && cd / && python -c "import torch; from groundingdino import _C; \
       import groundingdino.util.inference; \
       print('GroundingDINO CUDA extension: OK')"

# RAM++ likewise by path (_ascent_ram.pth in the live env). Installing
# `recognize-anything` from git would ship a different tree than the submodule
# the servers import.
RUN python -c "import site, pathlib; \
    pathlib.Path(site.getsitepackages()[0], '_ascent_ram.pth') \
    .write_text('/opt/ascent/third_party/recognize-anything\n')" \
    && python -c "import ram; print('RAM++ importable')"

RUN python -c "import torch, habitat_sim, transformers, numpy, cv2; \
    import lavis.models, mobile_sam, groundingdino.util.inference, ram; \
    from groundingdino import _C; \
    assert numpy.__version__.startswith('1.23'), numpy.__version__; \
    print('ascent env: all model packages import OK, numpy', numpy.__version__)"

# Exactly one opencv build may be installed: two of them abort at import with
# "CPU dispatcher tracer already initlized".
RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null; \
    python -c "import cv2; print('cv2', cv2.__version__)"

# ---------------------------------------------------------------------------
# 3. Runtime
# ---------------------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

# The ascent env's constraint must not follow the user into an interactive
# `pip install` in the habitat env.
ENV PIP_CONSTRAINT=

ENV NVIDIA_DRIVER_CAPABILITIES=all \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    HABITAT_COLLECTOR_ROOT=/habitat-data-collector \
    PYTHONPATH=/workspace/src:/habitat-data-collector \
    ASCENT_PYTHON=/opt/conda/envs/ascent/bin/python

# wgpu (the Rerun native viewer) probes every backend it is built with,
# including GLES-over-EGL, before picking one. That probe walks X11/GBM/Wayland
# and panics under the virtualized X server this container runs against, even
# though __EGL_VENDOR_LIBRARY_FILENAMES makes plain EGL fine for habitat-sim.
# Vulkan-only skips the probe; the GPU is still found as a Vulkan adapter.
ENV WGPU_BACKEND=vulkan

# GLVND dispatch + GLX for habitat-sim's bindings, and the five libraries winit
# dlopen()s at run time -- without them a GUI app fails with the misleading
# "Failed to load one of xlib's shared libraries" even though libX11 is present.
# libvulkan1 + mesa give wgpu a backend (and a llvmpipe fallback). This is only
# the CLIENT side: reaching a display also needs DISPLAY and the X socket from
# compose.yaml's x11 block.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    libxcursor1 libxrandr2 libxi6 libxkbcommon0 libxkbcommon-x11-0 \
    libxcomposite1 libxinerama1 libxtst6 \
    libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri \
    x11-utils xauth \
    && rm -rf /var/lib/apt/lists/*

RUN printf '\n# habitat env for `docker exec -it ... bash`\nif [ -z "$CONDA_DEFAULT_ENV" ]; then\n  source /opt/conda/etc/profile.d/conda.sh\n  conda activate habitat\nfi\n' >> /etc/bash.bashrc

COPY docker/modules/install_x11_opengl_vulkan.sh /tmp/install_x11_opengl_vulkan.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_x11_opengl_vulkan.sh && rm /tmp/install_x11_opengl_vulkan.sh

COPY docker/modules/install_ros2_humble.sh /tmp/install_ros2_humble.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_ros2_humble.sh && rm /tmp/install_ros2_humble.sh

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y sudo \
    && groupadd -g $USER_GID $USERNAME \
    && useradd -m -s /bin/bash -u $USER_UID -g $USER_GID -G sudo $USERNAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && chown -R $USER_UID:$USER_GID /opt/ascent \
    && rm -rf /var/lib/apt/lists/*

# `docker exec -it docker-nav-1 bash` -- the way everyone actually gets a shell
# here -- BYPASSES the ENTRYPOINT, so entrypoint.sh's `umask "${UMASK:-0002}"`
# never reaches those shells. They ran at the default 0022, every file written
# from them came out 0644, and a second host user sharing the mount could read
# the repository but not modify a single file in it. Set it where a bypassing
# shell still reads it: profile.d for login shells, bash.bashrc for the
# interactive non-login shell `docker exec ... bash` actually starts.
RUN printf 'umask %s\n' '002' > /etc/profile.d/00-shared-umask.sh \
    && chmod 0644 /etc/profile.d/00-shared-umask.sh \
    && printf '\n# Shared bind mount: new files must stay writable by the owning group.\numask 002\n' \
       >> /etc/bash.bashrc

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME \
    docker/modules/install_claude_code.sh /tmp/install_claude_code.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_claude_code.sh && rm /tmp/install_claude_code.sh

# Hardcoded rather than via ~/.bashrc: compose.yaml sets HOME=/tmp at run time,
# which both points $HOME/.local/bin at the wrong directory and is never sourced
# by the non-interactive `conda run` shell entrypoint.sh execs into.
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Model weights are NOT baked in (~5 GB, some need a HuggingFace token):
# `bash scripts/fetch_ascent_weights.sh` downloads them into the mounted
# workspace. See docs/SETUP.md.
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
