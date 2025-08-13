# Use NVIDIA CUDA base image compatible with your setup
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Update package lists and install essential tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    libxkbcommon-x11-dev \
    libglib2.0-0 && \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set working directory
WORKDIR /app

# Install Miniconda
ENV MINICONDA_VERSION=py310_23.1.0-1
ENV MINICONDA_INSTALLER_SCRIPT=Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh
ENV MINICONDA_PREFIX=/opt/conda

RUN wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT} && \
    chmod +x ${MINICONDA_INSTALLER_SCRIPT} && \
    ./${MINICONDA_INSTALLER_SCRIPT} -b -p ${MINICONDA_PREFIX} && \
    rm ${MINICONDA_INSTALLER_SCRIPT}

ENV PATH=$MINICONDA_PREFIX/bin:$PATH

# Initialize Conda so 'conda activate' works properly
RUN conda init bash

# Reset shell to default for later RUN commands
SHELL ["/bin/sh", "-c"]

# Install additional tools for model download
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface-hub for model download
RUN conda run -n hallo pip install huggingface-hub

# Create directory for pretrained models
RUN mkdir -p /app/pretrained_models

# Download main models from HuggingFace
RUN conda run -n hallo python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='fudan-generative-ai/hallo3', \
                  local_dir='/app/pretrained_models', \
                  local_dir_use_symlinks=False, \
                  revision='main')"

# Download additional models
# 1. Kim Vocal_2 model
RUN mkdir -p /app/pretrained_models/audio_separator && \
    wget -O /app/pretrained_models/audio_separator/Kim_Vocal_2.onnx \
    https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx

# 2. Face landmarker model
RUN mkdir -p /app/pretrained_models/face_analysis/models && \
    wget -O /app/pretrained_models/face_analysis/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Create Conda environment for the project
ENV CONDA_ENV_NAME=hallo
RUN conda create -n ${CONDA_ENV_NAME} python=3.10 pip -y

#Install PyTorch
RUN conda run -n hallo pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu122

# Install Python dependencies from requirements.txt
COPY .runpod/requirements.txt .
RUN conda run -n ${CONDA_ENV_NAME} pip install --no-cache-dir -r requirements.txt

# Install dlib via Conda-forge (more reliable)
RUN conda run -n ${CONDA_ENV_NAME} conda install -c conda-forge dlib -y && \
    conda clean --all -f -y

# Run the main script with the activated Conda environment
CMD ["conda", "run", "-n", "hallo", "python", "handler.py"]

