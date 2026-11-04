# syntax=docker/dockerfile:1.7
# CUDA 12.4 runtime for the PyTorch cu124 wheels and CTranslate2 CUDA runtime.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        build-essential \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /audio /saidas /models

# Install the CUDA-enabled PyTorch 2.5.1/cu124 wheels before application deps.
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 \
        torchaudio==2.5.1 \
        torchvision==0.20.1

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt \
    && python -m pip check

COPY diarize.py /app/diarize.py

VOLUME ["/audio", "/saidas", "/models"]
ENTRYPOINT ["python", "/app/diarize.py"]
CMD ["--check"]
