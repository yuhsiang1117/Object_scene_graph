FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# EGL/GL userland for headless habitat-sim rendering (driver libs injected by
# nvidia-container-toolkit at run time).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 unzip \
    libegl1 libgl1 libglvnd0 libx11-6 libxext6 \
    libjpeg-dev libpng-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
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
      scipy scikit-image opencv-python-headless pytest pytest-timeout \
      imageio matplotlib

# Headless EGL on NVIDIA
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
