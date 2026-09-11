FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Keep apt cache to speed up subsequent builds
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# EGL/GL userland for headless habitat-sim rendering (driver libs injected by
# nvidia-container-toolkit at run time).
# Added sudo and tmux here.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 unzip \
    libegl1 libgl1 libglvnd0 libx11-6 libxext6 \
    libjpeg-dev libpng-dev libglib2.0-0 \
    sudo tmux \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# habitat-sim 0.3.1 requires python 3.9
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
    scipy scikit-image scikit-learn opencv-python-headless pytest pytest-timeout \
    imageio matplotlib anthropic "rerun-sdk"

# Headless EGL on NVIDIA
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    PYTHONPATH=/workspace/src

# wgpu (used by the Rerun native viewer) probes every backend it's built
# with, including GLES-over-EGL, before picking one. That EGL probe walks
# the X11/GBM/Wayland platforms and panics (`Option::unwrap()` on `None` in
# khronos-egl) under the virtualized X server this container runs against,
# even though __EGL_VENDOR_LIBRARY_FILENAMES above makes plain EGL fine for
# habitat-sim. Pinning wgpu to vulkan-only skips that probe; the GPU is
# still picked up fine as a Vulkan adapter (verified against the RTX 4090).
ENV WGPU_BACKEND=vulkan

# GLVND OpenGL dispatch + GLX, required by habitat-sim bindings. Kept as a
# late layer so it reuses the (expensive) conda/torch layers above.
#
# The X11 client libs below are for GUI tools run inside the container -- the
# Rerun 3D scene-graph viewer (scripts/inspect_scene_graph.py). The base image
# ships libx11-6 but NOT the five libraries winit dlopen()s at runtime, so a
# GUI app fails with the misleading "Failed to load one of xlib's shared
# libraries" even though libX11 is present. libvulkan1 + mesa-vulkan-drivers
# give wgpu a backend (and a llvmpipe software fallback when the GPU has none).
#
# Note this only supplies the CLIENT side. Reaching a display also needs
# compose.yaml to pass DISPLAY and bind-mount the X socket -- see the x11
# block there. Without an X server the failure mode is the *other* message,
# "Failed to open connection to X server".
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    libopengl0 libglx0 \
    libxcursor1 libxrandr2 libxi6 libxkbcommon0 libxkbcommon-x11-0 \
    libxcomposite1 libxinerama1 libxtst6 \
    libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri \
    x11-utils xauth \
    && rm -rf /var/lib/apt/lists/*

# CLIP tokenizer used by YOLOE's text prompts (otherwise ultralytics
# AutoUpdate re-installs it in every ephemeral container).
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

RUN printf '\n# habitat env for `docker exec -it ... bash`\nif [ -z "$CONDA_DEFAULT_ENV" ]; then\n  source /opt/conda/etc/profile.d/conda.sh\n  conda activate habitat\nfi\n' >> /etc/bash.bashrc

# Install GUI debugging tools and configure Vulkan
COPY modules/install_x11_opengl_vulkan.sh /tmp/install_x11_opengl_vulkan.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_x11_opengl_vulkan.sh && rm /tmp/install_x11_opengl_vulkan.sh

COPY /entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y sudo \
    && groupadd -g $USER_GID $USERNAME \
    && useradd -m -s /bin/bash -u $USER_UID -g $USER_GID -G sudo $USERNAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME \
    modules/install_claude_code.sh /tmp/install_claude_code.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_claude_code.sh && rm /tmp/install_claude_code.sh

# Hardcode the install path rather than relying on ~/.bashrc: compose.yaml
# sets HOME=/tmp at runtime (for YOLO_CONFIG_DIR/etc.), which both points
# $HOME/.local/bin at the wrong directory and is never sourced anyway by the
# non-interactive `conda run` shell entrypoint.sh execs into.
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]