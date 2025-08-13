FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install system dependencies
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
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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

# Initialize Conda
RUN conda init bash

# Switch to bash login shell
SHELL ["/bin/bash", "--login", "-c"]

# Create Conda environment
ENV CONDA_ENV_NAME=hallo
RUN conda create -n ${CONDA_ENV_NAME} python=3.10 pip -y

# Install PyTorch
RUN conda run -n hallo pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu122

# Install PyAV with Conda (solution pour le problème de compatibilité)
RUN conda install -n hallo -c conda-forge av -y

# Install Python dependencies
COPY requirements.txt .
RUN conda run -n hallo pip install --no-cache-dir -r requirements.txt

# Install dlib
RUN conda run -n hallo conda install -c conda-forge dlib -y && \
    conda clean --all -f -y

# Install huggingface-hub
RUN conda run -n hallo pip install huggingface-hub

# Create models directory
RUN mkdir -p /app/pretrained_models

# Download main models
RUN conda run -n hallo python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='fudan-generative-ai/hallo3', \
                 local_dir='/app/pretrained_models', \
                 local_dir_use_symlinks=False, \
                 revision='main')"

# Download additional models
RUN mkdir -p /app/pretrained_models/audio_separator && \
    wget -O /app/pretrained_models/audio_separator/Kim_Vocal_2.onnx \
        https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx

RUN mkdir -p /app/pretrained_models/face_analysis/models && \
    wget -O /app/pretrained_models/face_analysis/models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Reset shell
SHELL ["/bin/sh", "-c"]

# Run the application
CMD ["conda", "run", "-n", "hallo", "python", "handler.py"]
